from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import os
import time
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipe = None
generated_image_data = None
model_path = "./NSFW-gen-v2"

# Track device and dtype used for loaded pipeline
pipe_device = None
pipe_dtype = None

# Auto-unload globals
last_activity = 0.0
UNLOAD_TIMEOUT = 30 * 60  # 30 minutes
_monitor_task = None
_pipe_lock = asyncio.Lock()

# Define the request schema
class ImageGenerationRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512

def _mark_activity():
    global last_activity
    last_activity = time.time()

# Choose best device
def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Load model function
def load_model():
    global pipe, pipe_device, pipe_dtype
    if pipe is not None:
        _mark_activity()
        return pipe

    logger.info("Loading model...")
    try:
        device = _select_device()
        logger.info(f"Selected device: {device}")

        # Pick dtype only when supported by target device
        selected_dtype = None
        if device == "cuda":
            selected_dtype = torch.float16
        elif device == "mps":
            # MPS works better with bfloat16 or float32
            selected_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else None

        # Load model with appropriate dtype
        if selected_dtype is not None:
            logger.info(f"Loading model with dtype: {selected_dtype}")
            pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=selected_dtype
            )
        else:
            logger.info("Loading model with default dtype (float32)")
            pipe = DiffusionPipeline.from_pretrained(model_path)

        # Move pipeline to selected device
        try:
            if device != "cpu":
                pipe.to(device)
                logger.info(f"Moved pipeline to {device}")
            else:
                logger.info("Using CPU device (no dtype conversion)")
        except Exception as e:
            logger.warning(f"Failed to move pipeline to {device}: {e}")

        # Store device and dtype info
        pipe_device = device
        pipe_dtype = selected_dtype

        _mark_activity()
        logger.info("Model loaded successfully!")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

async def _async_unload_model():
    """
    Unload model to free memory. Safe to call concurrently.
    """
    global pipe, generated_image_data, last_activity, pipe_device, pipe_dtype
    async with _pipe_lock:
        if pipe is None:
            return False
        try:
            logger.info("Unloading model to free memory...")
            
            # Only try to move to CPU if the pipeline wasn't loaded with float16/bfloat16
            # or if we're already on CPU
            if pipe_dtype is None or pipe_device == "cpu":
                try:
                    pipe.to("cpu")
                    logger.info("Moved pipeline to CPU before deletion")
                except Exception as e:
                    logger.warning(f"Failed to move pipeline to CPU: {e}")
            else:
                logger.info(f"Skipping CPU move for {pipe_dtype} pipeline to avoid errors")

            del pipe
            generated_image_data = None
        except Exception as e:
            logger.warning(f"Error while deleting pipe: {e}")
        finally:
            pipe = None
            pipe_device = None
            pipe_dtype = None
            
            # Clear cache based on device
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
                except Exception:
                    pass
            
            gc.collect()
            last_activity = 0.0
            logger.info("Model unloaded.")
            return True

async def _monitor_unload():
    global _monitor_task, last_activity
    logger.info("Starting model unload monitor task")
    try:
        while True:
            await asyncio.sleep(30)  # check interval
            if pipe is None:
                continue
            if last_activity == 0:
                continue
            idle = time.time() - last_activity
            if idle >= UNLOAD_TIMEOUT:
                logger.info(f"No activity for {int(idle)}s (>= {UNLOAD_TIMEOUT}s). Unloading model.")
                await _async_unload_model()
    except asyncio.CancelledError:
        logger.info("Model unload monitor cancelled")
        raise
    except Exception as e:
        logger.error(f"Model unload monitor error: {e}")

# Startup/shutdown hooks
@app.on_event("startup")
async def _start_monitor():
    global _monitor_task
    if _monitor_task is None:
        _monitor_task = asyncio.create_task(_monitor_unload())

@app.on_event("shutdown")
async def _stop_monitor():
    global _monitor_task
    if _monitor_task:
        _monitor_task.cancel()
        try:
            await _monitor_task
        except asyncio.CancelledError:
            pass
        _monitor_task = None
    try:
        await _async_unload_model()
    except Exception:
        pass

# WebSocket endpoint for progress updates
@app.websocket("/progress")
async def progress_endpoint(websocket: WebSocket):
    global generated_image_data, pipe
    await websocket.accept()

    if pipe is None:
        try:
            await websocket.send_text("Loading model...")
            load_model()
            await websocket.send_text("Model loaded successfully!")
        except Exception as e:
            await websocket.send_text(f"Error loading model: {str(e)}")
            await websocket.close()
            return

    _mark_activity()

    # Progress tracking variables
    websocket_active = True
    current_step = 0

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        num_inference_steps = data.get("num_inference_steps", 20)
        guidance_scale = data.get("guidance_scale", 7.5)
        width = data.get("width", 512)
        height = data.get("height", 512)

        ws_loop = asyncio.get_event_loop()

        def progress_callback(pipe_obj, step_index, timestep, callback_kwargs):
            nonlocal current_step, websocket_active

            if not websocket_active:
                return callback_kwargs

            current_step += 1
            progress_percentage = int((current_step / num_inference_steps) * 100)

            try:
                if websocket_active and getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                    asyncio.run_coroutine_threadsafe(
                        send_progress_safely(websocket, progress_percentage),
                        ws_loop
                    )
                else:
                    websocket_active = False
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
                websocket_active = False

            return callback_kwargs

        async def send_progress_safely(ws, progress):
            nonlocal websocket_active
            try:
                if getattr(ws, "client_state", None) and ws.client_state.name == "CONNECTED":
                    await ws.send_text(f"Progress: {progress}%")
                    logger.info(f"Sent progress: {progress}%")
                else:
                    websocket_active = False
            except Exception as e:
                logger.error(f"Failed to send progress: {e}")
                websocket_active = False

        def generate_image():
            _mark_activity()
            return pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                callback_on_step_end=progress_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            ).images[0]

        _mark_activity()
        image = await asyncio.get_event_loop().run_in_executor(None, generate_image)
        _mark_activity()

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        generated_image_data = img_byte_arr.getvalue()

        if websocket_active and getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
            await websocket.send_text("Image generation completed!")
            logger.info("Image generation completed!")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
        websocket_active = False
    except Exception as e:
        logger.error(f"Generation error: {e}")
        websocket_active = False
        try:
            if getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                await websocket.send_text(f"Error: {str(e)}")
        except:
            pass
    finally:
        websocket_active = False
        try:
            if getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                await websocket.close()
        except:
            pass

# REST endpoint to fetch the generated image
@app.post("/generate-image")
async def get_generated_image(request: ImageGenerationRequest):
    global generated_image_data
    if generated_image_data is None:
        raise HTTPException(status_code=404, detail="No image has been generated yet")
    _mark_activity()
    return StreamingResponse(
        BytesIO(generated_image_data),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=generated_image.png"}
    )

# Direct generation endpoint (alternative to WebSocket)
@app.post("/generate-direct")
async def generate_direct(request: ImageGenerationRequest):
    global pipe, generated_image_data
    if pipe is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    _mark_activity()

    try:
        def generate_image():
            _mark_activity()
            return pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
            ).images[0]

        _mark_activity()
        image = await asyncio.get_event_loop().run_in_executor(None, generate_image)
        _mark_activity()

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        generated_image_data = img_byte_arr.getvalue()
        _mark_activity()

        return StreamingResponse(
            BytesIO(generated_image_data),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )

    except Exception as e:
        logger.error(f"Direct generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    global pipe, pipe_device, pipe_dtype
    model_loaded = pipe is not None
    return {
        "message": "Text-to-Image API is running",
        "model_loaded": model_loaded,
        "device": pipe_device if model_loaded else None,
        "dtype": str(pipe_dtype) if model_loaded and pipe_dtype else None
    }

# Model info endpoint
@app.get("/model-info")
async def model_info():
    global pipe, pipe_device, pipe_dtype
    if pipe is None:
        return {
            "model_name": "NSFW-gen-v2",
            "model_loaded": False,
            "message": "Model not loaded yet. It will be loaded on first request."
        }
    
    return {
        "model_name": "NSFW-gen-v2",
        "model_loaded": True,
        "device": pipe_device,
        "dtype": str(pipe_dtype) if pipe_dtype else "float32",
        "model_path": model_path
    }

# Load model endpoint (for manual loading)
@app.post("/load-model")
async def load_model_endpoint():
    global pipe
    try:
        if pipe is None:
            load_model()
            return {"message": "Model loaded successfully"}
        else:
            _mark_activity()
            return {"message": "Model already loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Manual unload endpoint
@app.post("/unload-model")
async def unload_model_endpoint():
    success = await _async_unload_model()
    if success:
        return {"message": "Model unloaded"}
    else:
        return {"message": "Model was not loaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)