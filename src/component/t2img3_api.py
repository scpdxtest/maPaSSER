from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import os
import time
import gc
import argparse
import sys
from PIL import Image
import numpy as np

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

# Set environment variable to help with MPS memory management
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Global variables - will be set by command line arguments
lora_name = "Lustly"
lora_file = None
lora_path = None
model_path = None

# Pipeline globals
pipe = None
pipe_device = None
pipe_dtype = None
generated_image_data = None

# Auto-unload globals
last_activity = 0.0
UNLOAD_TIMEOUT = 30 * 60  # 30 minutes
_monitor_task = None
_pipe_lock = asyncio.Lock()

# Define the request schema
class ImageGenerationRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    width: int = 768
    height: int = 768
    lora_weight: float = 1.0  # Allow adjusting LoRA weight

def _mark_activity():
    global last_activity
    last_activity = time.time()

def _select_device():
    """Select the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def _sanitize_image(image):
    """
    Sanitize the generated image to ensure it contains valid pixel values.
    Fixes the RuntimeWarning: invalid value encountered in cast
    """
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image).astype(np.float32)
        else:
            image_array = np.asarray(image).astype(np.float32)

        # Check for invalid values (NaN, inf, -inf)
        if not np.all(np.isfinite(image_array)):
            logger.warning("Detected invalid values (NaN/inf) in generated image. Sanitizing...")
            # Replace NaN with 0, positive infinity with 255, negative infinity with 0
            image_array = np.nan_to_num(image_array, nan=0.0, posinf=255.0, neginf=0.0)

        # Ensure values are in correct range
        # If values are in [0,1] range, scale to [0,255]
        if image_array.max() <= 1.0 and image_array.min() >= 0.0:
            image_array = image_array * 255.0
        
        # Clip to valid range and convert to uint8
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]  # Remove alpha channel

        # Convert back to PIL Image
        sanitized_image = Image.fromarray(image_array, mode='RGB')
        
        # Verify the image is valid
        if sanitized_image.size[0] == 0 or sanitized_image.size[1] == 0:
            raise ValueError("Generated image has zero dimensions")
            
        return sanitized_image
        
    except Exception as e:
        logger.error(f"Failed to sanitize image: {e}")
        # Return a fallback image (white square)
        fallback_array = np.full((512, 512, 3), 255, dtype=np.uint8)
        return Image.fromarray(fallback_array, mode='RGB')

# Load model function
def load_model():
    global pipe, pipe_device, pipe_dtype, model_path, lora_path, lora_file, lora_name
    if pipe is not None:
        _mark_activity()
        return pipe

    if model_path is None:
        raise ValueError("Model path not set. Please provide --model-path argument.")

    logger.info(f"Loading model from: {model_path}")
    device = _select_device()
    logger.info(f"Selected device: {device}")

    # Choose appropriate dtype based on device
    if device == "cuda":
        selected_dtype = torch.float16
    elif device == "mps":
        # Use float32 for MPS to avoid precision issues
        selected_dtype = torch.float32
    else:  # CPU
        selected_dtype = torch.float32

    try:
        logger.info(f"Loading model with dtype: {selected_dtype}")
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=selected_dtype,
            use_safetensors=True
        )
        
        # Enable memory optimizations
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        
        if hasattr(pipe, "enable_model_cpu_offload") and device != "cpu":
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")

        # Move to device
        try:
            pipe.to(device)
            logger.info(f"Moved pipeline to {device}")
        except Exception as e:
            logger.warning(f"Failed to move pipe to '{device}': {e}, continuing with default device")

        # Load LoRA if available and paths are provided
        # Replace the LoRA loading section with this more robust version:
        if lora_path and lora_file:
            try:
                lora_full_path = os.path.join(lora_path, lora_file)
                if os.path.exists(lora_full_path):
                    logger.info(f"Attempting to load LoRA from {lora_path}/{lora_file}")
                    
                    # Try loading with error handling
                    try:
                        pipe.load_lora_weights(
                            lora_path,
                            weight_name=lora_file,
                            adapter_name=lora_name
                        )
                        
                        # Verify the adapter was loaded
                        if hasattr(pipe, 'get_list_adapters'):
                            adapters = pipe.get_list_adapters()
                            if lora_name in adapters:
                                pipe.set_adapters([lora_name], adapter_weights=[1.0])
                                logger.info(f"LoRA adapter '{lora_name}' loaded and activated successfully")
                            else:
                                logger.warning(f"LoRA adapter '{lora_name}' not found in loaded adapters: {adapters}")
                        else:
                            logger.warning("Cannot verify LoRA loading - get_list_adapters not available")
                            
                    except Exception as lora_error:
                        logger.warning(f"LoRA incompatible with current model: {lora_error}")
                        logger.info("Continuing without LoRA - model will work with base capabilities")
                else:
                    logger.warning(f"LoRA file not found at {lora_full_path}")
            except Exception as e:
                logger.warning(f"LoRA loading failed: {e}. Continuing without LoRA...")
                
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
            # Safely move to CPU before deletion
            try:
                if pipe_device != "cpu":
                    pipe.to("cpu")
                    logger.info("Moved pipeline to CPU before deletion")
            except Exception as e:
                logger.warning(f"Failed to move to CPU: {e}")
            
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
                torch.cuda.empty_cache()
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
            await websocket.send_json({"type": "text", "message": "Loading model..."})
            load_model()
            await websocket.send_json({"type": "text", "message": "Model loaded successfully!"})
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
            return

    _mark_activity()

    # Progress tracking variables
    websocket_active = True
    start_time = None

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        num_inference_steps = data.get("num_inference_steps", 25)
        guidance_scale = data.get("guidance_scale", 7.5)
        width = data.get("width", 768)
        height = data.get("height", 768)
        lora_weight = data.get("lora_weight", 1.0)

        # Ensure dimensions are divisible by 8 (required by most diffusion models)
        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(f"Starting generation: {width}x{height}, steps: {num_inference_steps}, guidance: {guidance_scale}")

        try:
            if hasattr(pipe, 'get_list_adapters') and lora_name in pipe.get_list_adapters():
                pipe.set_adapters([lora_name], adapter_weights=[lora_weight])
                logger.info(f"Set LoRA weight to {lora_weight}")
        except Exception as e:
            logger.warning(f"Could not adjust LoRA weight: {e}")

        ws_loop = asyncio.get_event_loop()

        def format_eta(seconds):
            if seconds is None or seconds <= 0:
                return "00:00"
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        def progress_callback(pipe_obj, step_index, timestep, callback_kwargs):
            nonlocal websocket_active, start_time

            if not websocket_active:
                return callback_kwargs

            steps_done = step_index + 1
            progress_percentage = int((steps_done / max(1, num_inference_steps)) * 100)

            if start_time is None:
                start_time = time.time()

            elapsed = time.time() - start_time
            avg_step = elapsed / steps_done if steps_done > 0 else None
            remaining_steps = max(0, num_inference_steps - steps_done)
            eta_seconds = remaining_steps * avg_step if avg_step is not None else None

            payload = {
                "type": "progress",
                "progress": progress_percentage,
                "steps_done": steps_done,
                "steps_total": num_inference_steps,
                "eta_seconds": eta_seconds,
                "eta_str": format_eta(eta_seconds) if eta_seconds is not None else None,
            }

            try:
                if websocket_active and getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                    asyncio.run_coroutine_threadsafe(send_progress_safely(websocket, payload), ws_loop)
                else:
                    websocket_active = False
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
                websocket_active = False

            return callback_kwargs

        async def send_progress_safely(ws, payload):
            nonlocal websocket_active
            try:
                if getattr(ws, "client_state", None) and ws.client_state.name == "CONNECTED":
                    await ws.send_json(payload)
                else:
                    websocket_active = False
            except Exception as e:
                logger.error(f"Failed to send progress: {e}")
                websocket_active = False

        def generate_image():
            _mark_activity()
            try:
                # Generate with appropriate precision context
                if pipe_device == "cuda":
                    with torch.cuda.amp.autocast():
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            callback_on_step_end=progress_callback,
                            callback_on_step_end_tensor_inputs=["latents"]
                        )
                else:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        callback_on_step_end=progress_callback,
                        callback_on_step_end_tensor_inputs=["latents"]
                    )
                
                raw_image = result.images[0]
                logger.info(f"Generated image type: {type(raw_image)}, mode: {getattr(raw_image, 'mode', 'unknown')}")
                
                # Sanitize the image to fix any invalid values
                sanitized_image = _sanitize_image(raw_image)
                logger.info(f"Image sanitized successfully: {sanitized_image.size}")
                
                return sanitized_image
                
            except Exception as e:
                logger.error(f"Error during image generation: {e}")
                raise

        _mark_activity()
        image = await asyncio.get_event_loop().run_in_executor(None, generate_image)
        _mark_activity()

        # Save the sanitized image
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG", optimize=True)
        img_byte_arr.seek(0)
        generated_image_data = img_byte_arr.getvalue()

        if websocket_active and getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
            await websocket.send_json({"type": "complete"})
            logger.info("Image generation completed!")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
        websocket_active = False
    except Exception as e:
        logger.error(f"Generation error: {e}")
        websocket_active = False
        try:
            if getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                await websocket.send_json({"type": "error", "message": str(e)})
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
        try:
            if hasattr(pipe, 'get_list_adapters') and lora_name in pipe.get_list_adapters():
                pipe.set_adapters([lora_name], adapter_weights=[request.lora_weight])
                logger.info(f"Set LoRA weight to {request.lora_weight}")
        except Exception as e:
            logger.warning(f"Could not adjust LoRA weight: {e}")

        # Ensure dimensions are divisible by 8
        width = (request.width // 8) * 8
        height = (request.height // 8) * 8

        def generate_image():
            _mark_activity()
            try:
                # Generate with appropriate precision context
                if pipe_device == "cuda":
                    with torch.cuda.amp.autocast():
                        result = pipe(
                            prompt=request.prompt,
                            negative_prompt=request.negative_prompt,
                            num_inference_steps=request.num_inference_steps,
                            guidance_scale=request.guidance_scale,
                            width=width,
                            height=height,
                        )
                else:
                    result = pipe(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        width=width,
                        height=height,
                    )
                
                raw_image = result.images[0]
                logger.info(f"Generated image type: {type(raw_image)}, mode: {getattr(raw_image, 'mode', 'unknown')}")
                
                # Sanitize the image to fix any invalid values
                sanitized_image = _sanitize_image(raw_image)
                logger.info(f"Image sanitized successfully: {sanitized_image.size}")
                
                return sanitized_image
                
            except Exception as e:
                logger.error(f"Error during direct generation: {e}")
                raise

        _mark_activity()
        image = await asyncio.get_event_loop().run_in_executor(None, generate_image)
        _mark_activity()

        # Save the sanitized image
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG", optimize=True)
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
    global pipe, pipe_device, pipe_dtype, model_path, lora_path, lora_file
    model_loaded = pipe is not None
    adapters = []
    if model_loaded and hasattr(pipe, 'get_list_adapters'):
        try:
            adapters = pipe.get_list_adapters()
        except:
            pass
    return {
        "message": "Text-to-Image API is running",
        "model_path": model_path,
        "model": os.path.basename(model_path) if model_path else "No model path set",
        "model_loaded": model_loaded,
        "lora_path": lora_path,
        "lora_file": lora_file,
        "lora_loaded": lora_name in adapters,
        "device": pipe_device,
        "dtype": str(pipe_dtype) if pipe_dtype else None
    }

# Model info endpoint
@app.get("/model-info")
async def model_info():
    global pipe, pipe_device, pipe_dtype, model_path, lora_path, lora_file
    if pipe is None:
        return {
            "model_name": os.path.basename(model_path) if model_path else "No model path set",
            "model_path": model_path,
            "lora_path": lora_path,
            "lora_file": lora_file,
            "model_loaded": False,
            "message": "Model not loaded yet. It will be loaded on first request."
        }
    try:
        adapters = pipe.get_list_adapters() if hasattr(pipe, 'get_list_adapters') else []
        return {
            "model_name": os.path.basename(model_path),
            "model_path": model_path,
            "lora_path": lora_path,
            "lora_file": lora_file,
            "model_loaded": True,
            "device": pipe_device,
            "dtype": str(pipe_dtype) if pipe_dtype else "unknown",
            "loaded_adapters": adapters
        }
    except Exception as e:
        return {
            "model_name": os.path.basename(model_path) if model_path else "Unknown",
            "model_loaded": True,
            "error": str(e)
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Text-to-Image API Server with LoRA support")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the diffusion model directory"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to the LoRA weights directory (optional)"
    )
    parser.add_argument(
        "--lora-file",
        type=str,
        help="LoRA filename (e.g., lora.safetensors) (optional)"
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default="Lustly",
        help="Name for the LoRA adapter (default: Lustly)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server to (default: 8001)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Auto-unload timeout in minutes (default: 30)"
    )
    return parser.parse_args()

def validate_model_path(path):
    """Validate that the model path exists and is accessible"""
    if not os.path.exists(path):
        logger.error(f"Model path does not exist: {path}")
        return False
    
    if not os.path.isdir(path):
        logger.error(f"Model path is not a directory: {path}")
        return False
    
    # Check for common model files
    expected_files = ["config.json", "model_index.json"]
    has_expected_file = False
    
    for file in expected_files:
        if os.path.exists(os.path.join(path, file)):
            has_expected_file = True
            break
    
    if not has_expected_file:
        logger.warning(f"Model path may not contain a valid diffusion model: {path}")
        logger.warning(f"Expected to find one of: {expected_files}")
    
    return True

def validate_lora_config(lora_path_arg, lora_file_arg):
    """Validate LoRA configuration"""
    if lora_path_arg and lora_file_arg:
        if not os.path.exists(lora_path_arg):
            logger.error(f"LoRA path does not exist: {lora_path_arg}")
            return False
        
        if not os.path.isdir(lora_path_arg):
            logger.error(f"LoRA path is not a directory: {lora_path_arg}")
            return False
        
        lora_full_path = os.path.join(lora_path_arg, lora_file_arg)
        if not os.path.exists(lora_full_path):
            logger.error(f"LoRA file does not exist: {lora_full_path}")
            return False
        
        logger.info(f"LoRA configuration validated: {lora_full_path}")
        
        return True
    elif lora_path_arg or lora_file_arg:
        logger.error("Both --lora-path and --lora-file must be provided together")
        return False
    else:
        logger.info("No LoRA configuration provided")
        return True

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate model path
    if not validate_model_path(args.model_path):
        logger.error("Invalid model path provided. Exiting.")
        sys.exit(1)
    
    # Validate LoRA configuration
    if not validate_lora_config(args.lora_path, args.lora_file):
        logger.error("Invalid LoRA configuration. Exiting.")
        sys.exit(1)
    
    # Set global variables
    model_path = args.model_path
    lora_path = args.lora_path
    lora_file = args.lora_file
    lora_name = args.lora_name
    
    # Set global timeout
    UNLOAD_TIMEOUT = args.timeout * 60  # Convert minutes to seconds
    
    logger.info(f"Starting Text-to-Image API server")
    logger.info(f"Model path: {model_path}")
    if lora_path and lora_file:
        logger.info(f"LoRA path: {lora_path}")
        logger.info(f"LoRA file: {lora_file}")
        logger.info(f"LoRA name: {lora_name}")
    else:
        logger.info("No LoRA configuration provided")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Auto-unload timeout: {args.timeout} minutes")
    
    import uvicorn
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)