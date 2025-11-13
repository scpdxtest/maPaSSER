from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image, FluxImg2ImgPipeline, DiffusionPipeline
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
import base64
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
lora_name = ""
lora_file = None
lora_path = None
model_path = None

# Pipeline globals
pipe_t2i = None  # Text-to-image pipeline
pipe_i2i = None  # Image-to-image pipeline
pipe_device = None
pipe_dtype = None
generated_image_data = None

# Auto-unload globals
last_activity = 0.0
UNLOAD_TIMEOUT = 30 * 60  # 30 minutes
_monitor_task = None
_pipe_lock = asyncio.Lock()
_generation_in_progress = False
_generation_lock = asyncio.Lock()

# Define the request schemas
class ImageGenerationRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    width: int = 768
    height: int = 768
    lora_weight: float = 1.0

class ImageEditRequest(BaseModel):
    image_base64: str
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    strength: float = 0.7
    lora_weight: float = 1.0

def _mark_activity():
    global last_activity
    last_activity = time.time()

def _start_generation():
    global _generation_in_progress
    _generation_in_progress = True
    _mark_activity()

def _end_generation():
    global _generation_in_progress
    _generation_in_progress = False
    _mark_activity()

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
    """
    try:
        if isinstance(image, Image.Image):
            image_array = np.array(image).astype(np.float32)
        else:
            image_array = np.asarray(image).astype(np.float32)

        if not np.all(np.isfinite(image_array)):
            logger.warning("Detected invalid values (NaN/inf) in generated image. Sanitizing...")
            image_array = np.nan_to_num(image_array, nan=0.0, posinf=255.0, neginf=0.0)

        if image_array.max() <= 1.0 and image_array.min() >= 0.0:
            image_array = image_array * 255.0
        
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]

        sanitized_image = Image.fromarray(image_array, mode='RGB')
        
        if sanitized_image.size[0] == 0 or sanitized_image.size[1] == 0:
            raise ValueError("Generated image has zero dimensions")
            
        return sanitized_image
        
    except Exception as e:
        logger.error(f"Failed to sanitize image: {e}")
        fallback_array = np.full((512, 512, 3), 255, dtype=np.uint8)
        return Image.fromarray(fallback_array, mode='RGB')

def load_t2i_model():
    """Load text-to-image pipeline with LoRA support"""
    global pipe_t2i, pipe_device, pipe_dtype, model_path, lora_path, lora_file, lora_name
    if pipe_t2i is not None:
        _mark_activity()
        return pipe_t2i

    if model_path is None:
        raise ValueError("Model path not set. Please provide --model-path argument.")

    logger.info(f"Loading T2I model from: {model_path}")
    device = _select_device()
    logger.info(f"Selected device: {device}")

    # Choose appropriate dtype based on device
    if device == "cuda":
        selected_dtype = torch.float16
    elif device == "mps":
        selected_dtype = torch.float32
    else:  # CPU
        selected_dtype = torch.float32

    try:
        logger.info(f"Loading T2I model with dtype: {selected_dtype}")
        pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=selected_dtype,
            use_safetensors=True
        )
        
        # Enable memory optimizations
        if hasattr(pipe_t2i, "enable_attention_slicing"):
            pipe_t2i.enable_attention_slicing()
            logger.info("Enabled attention slicing for T2I")
        
        if hasattr(pipe_t2i, "enable_model_cpu_offload") and device != "cpu":
            pipe_t2i.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload for T2I")

        # Move to device
        try:
            pipe_t2i.to(device)
            logger.info(f"Moved T2I pipeline to {device}")
        except Exception as e:
            logger.warning(f"Failed to move T2I pipe to '{device}': {e}")

        # Load LoRA if available
        if lora_path and lora_file:
            try:
                lora_full_path = os.path.join(lora_path, lora_file)
                if os.path.exists(lora_full_path):
                    logger.info(f"Loading LoRA for T2I from {lora_path}/{lora_file}")
                    try:
                        pipe_t2i.load_lora_weights(
                            lora_path,
                            weight_name=lora_file,
                            adapter_name=lora_name
                        )
                        
                        if hasattr(pipe_t2i, 'get_list_adapters'):
                            adapters = pipe_t2i.get_list_adapters()
                            if lora_name in adapters:
                                pipe_t2i.set_adapters([lora_name], adapter_weights=[1.0])
                                logger.info(f"T2I LoRA adapter '{lora_name}' loaded successfully")
                            else:
                                logger.warning(f"T2I LoRA adapter '{lora_name}' not found in adapters: {adapters}")
                    except Exception as lora_error:
                        logger.warning(f"T2I LoRA incompatible: {lora_error}")
                else:
                    logger.warning(f"T2I LoRA file not found at {lora_full_path}")
            except Exception as e:
                logger.warning(f"T2I LoRA loading failed: {e}")

        pipe_device = device
        pipe_dtype = selected_dtype
        _mark_activity()
        logger.info("T2I Model loaded successfully!")
        return pipe_t2i

    except Exception as e:
        logger.error(f"Failed to load T2I model: {e}")
        raise e

def load_i2i_model():
    """Load image-to-image pipeline with LoRA support"""
    global pipe_i2i, pipe_device, pipe_dtype, model_path, lora_path, lora_file, lora_name
    if pipe_i2i is not None:
        _mark_activity()
        return pipe_i2i

    if model_path is None:
        raise ValueError("Model path not set. Please provide --model-path argument.")

    # ADDED: Unload T2I model before loading I2I to save memory
    if pipe_t2i is not None:
        logger.info("Unloading T2I model to free memory for I2I loading...")
        try:
            if pipe_device and pipe_device != "cpu":
                pipe_t2i.to("cpu")
            del pipe_t2i
            pipe_t2i = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("T2I model unloaded successfully")
        except Exception as e:
            logger.warning(f"Error unloading T2I model: {e}")
    
    logger.info(f"Loading I2I model from: {model_path}")
    device = _select_device() if pipe_device is None else pipe_device
    
    selected_dtype = pipe_dtype if pipe_dtype is not None else (
        torch.float16 if device == "cuda" else torch.float32
    )

    try:
        logger.info(f"Loading I2I model with dtype: {selected_dtype}")
        
        # Try to load FLUX img2img pipeline first
        try:
            pipe_i2i = FluxImg2ImgPipeline.from_pretrained(
                model_path,
                torch_dtype=selected_dtype,
                use_safetensors=True
            )
            logger.info("Loaded FLUX I2I pipeline")
        except Exception as e:
            logger.warning(f"Failed to load FLUX I2I pipeline: {e}")
            # Try generic pipeline as fallback
            try:
                pipe_i2i = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=selected_dtype,
                    use_safetensors=True
                )
                logger.info("Loaded generic I2I pipeline as fallback")
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Model does not support image editing: {e2}")
        
        # Enable memory optimizations
        if hasattr(pipe_i2i, "enable_attention_slicing"):
            pipe_i2i.enable_attention_slicing()
            logger.info("Enabled attention slicing for I2I")
        
        if hasattr(pipe_i2i, "enable_model_cpu_offload") and device != "cpu":
            pipe_i2i.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload for I2I")

        # Move to device
        try:
            pipe_i2i.to(device)
            logger.info(f"Moved I2I pipeline to {device}")
        except Exception as e:
            logger.warning(f"Failed to move I2I pipe to '{device}': {e}")

        # Load LoRA if available
        if lora_path and lora_file:
            try:
                lora_full_path = os.path.join(lora_path, lora_file)
                if os.path.exists(lora_full_path):
                    logger.info(f"Loading LoRA for I2I from {lora_path}/{lora_file}")
                    try:
                        pipe_i2i.load_lora_weights(
                            lora_path,
                            weight_name=lora_file,
                            adapter_name=lora_name
                        )
                        
                        if hasattr(pipe_i2i, 'get_list_adapters'):
                            adapters = pipe_i2i.get_list_adapters()
                            if lora_name in adapters:
                                pipe_i2i.set_adapters([lora_name], adapter_weights=[1.0])
                                logger.info(f"I2I LoRA adapter '{lora_name}' loaded successfully")
                            else:
                                logger.warning(f"I2I LoRA adapter '{lora_name}' not found in adapters: {adapters}")
                    except Exception as lora_error:
                        logger.warning(f"I2I LoRA incompatible: {lora_error}")
                else:
                    logger.warning(f"I2I LoRA file not found at {lora_full_path}")
            except Exception as e:
                logger.warning(f"I2I LoRA loading failed: {e}")

        if pipe_device is None:
            pipe_device = device
            pipe_dtype = selected_dtype
        
        _mark_activity()
        logger.info("I2I Model loaded successfully!")
        return pipe_i2i

    except Exception as e:
        logger.error(f"Failed to load I2I model: {e}")
        raise e

async def _async_unload_models():
    """Unload models to free memory"""
    global pipe_t2i, pipe_i2i, generated_image_data, last_activity, pipe_device, pipe_dtype, _generation_in_progress
    async with _pipe_lock:
        if _generation_in_progress:
            logger.info("Generation in progress, skipping unload")
            return False
            
        models_unloaded = False
        
        if pipe_t2i is not None:
            try:
                logger.info("Unloading T2I model...")
                if pipe_dtype is None or pipe_device == "cpu":
                    try:
                        pipe_t2i.to("cpu")
                        logger.info("Moved T2I pipeline to CPU before delete")
                    except Exception as e:
                        logger.warning(f"Move T2I to CPU failed: {e}")
                del pipe_t2i
                models_unloaded = True
            except Exception as e:
                logger.warning(f"Error deleting T2I pipe: {e}")
            finally:
                pipe_t2i = None
        
        if pipe_i2i is not None:
            try:
                logger.info("Unloading I2I model...")
                if pipe_dtype is None or pipe_device == "cpu":
                    try:
                        pipe_i2i.to("cpu")
                        logger.info("Moved I2I pipeline to CPU before delete")
                    except Exception as e:
                        logger.warning(f"Move I2I to CPU failed: {e}")
                del pipe_i2i
                models_unloaded = True
            except Exception as e:
                logger.warning(f"Error deleting I2I pipe: {e}")
            finally:
                pipe_i2i = None
        
        if models_unloaded:
            generated_image_data = None
            pipe_device = None
            pipe_dtype = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()
            last_activity = 0.0
            logger.info("Models unloaded")
            return True
        
        return False

async def _monitor_unload():
    global _monitor_task, last_activity, _generation_in_progress
    logger.info("Starting model unload monitor task")
    try:
        while True:
            await asyncio.sleep(30)
            if pipe_t2i is None and pipe_i2i is None:
                continue
            if last_activity == 0:
                continue
            if _generation_in_progress:
                continue
            idle = time.time() - last_activity
            if idle >= UNLOAD_TIMEOUT:
                logger.info(f"No activity for {int(idle)}s (>= {UNLOAD_TIMEOUT}s). Unloading models.")
                await _async_unload_models()
    except asyncio.CancelledError:
        logger.info("Monitor cancelled")
        raise
    except Exception as e:
        logger.error(f"Monitor error: {e}")

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
        await _async_unload_models()
    except Exception:
        pass

# WebSocket endpoint for text-to-image generation (alias for /progress)
@app.websocket("/generate")
async def generate_endpoint(websocket: WebSocket):
    """WebSocket endpoint for text-to-image generation (compatible with ImageEditChat.js)"""
    await progress_endpoint(websocket)

# WebSocket endpoint for text-to-image generation
@app.websocket("/progress")
async def progress_endpoint(websocket: WebSocket):
    global generated_image_data, pipe_t2i
    await websocket.accept()

    if _generation_in_progress:
        await websocket.send_json({"type": "error", "message": "Another generation is already in progress. Please wait."})
        await websocket.close()
        return

    async with _generation_lock:
        if pipe_t2i is None:
            try:
                await websocket.send_json({"type": "text", "message": "Loading T2I model..."})
                load_t2i_model()
                await websocket.send_json({"type": "text", "message": "T2I Model loaded successfully!"})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                await websocket.close()
                return

        _start_generation()
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

            # Ensure dimensions are divisible by 8
            width = (width // 8) * 8
            height = (height // 8) * 8

            logger.info(f"Starting T2I generation: {width}x{height}, steps: {num_inference_steps}, guidance: {guidance_scale}")

            # Set LoRA weight
            try:
                if hasattr(pipe_t2i, 'get_list_adapters') and lora_name in pipe_t2i.get_list_adapters():
                    pipe_t2i.set_adapters([lora_name], adapter_weights=[lora_weight])
                    logger.info(f"Set T2I LoRA weight to {lora_weight}")
            except Exception as e:
                logger.warning(f"Could not adjust T2I LoRA weight: {e}")

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
                    logger.error(f"Error in T2I progress callback: {e}")
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
                    logger.error(f"Failed to send T2I progress: {e}")
                    websocket_active = False

            def generate_image():
                _mark_activity()
                try:
                    if pipe_device == "cuda":
                        with torch.cuda.amp.autocast():
                            result = pipe_t2i(
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
                        result = pipe_t2i(
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
                    return _sanitize_image(raw_image)
                except Exception as e:
                    logger.error(f"Error during T2I generation: {e}")
                    raise

            _mark_activity()
            image = await asyncio.get_event_loop().run_in_executor(None, generate_image)
            _mark_activity()

            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG", optimize=True)
            img_byte_arr.seek(0)
            generated_image_data = img_byte_arr.getvalue()

            if websocket_active and getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                await websocket.send_json({"type": "complete"})
                logger.info("T2I generation completed!")

        except WebSocketDisconnect:
            logger.info("T2I WebSocket disconnected by client")
            websocket_active = False
        except Exception as e:
            logger.error(f"T2I Generation error: {e}")
            websocket_active = False
            try:
                if getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({"type": "error", "message": str(e)})
            except:
                pass
        finally:
            _end_generation()
            websocket_active = False
            try:
                if getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                    await websocket.close()
            except:
                pass

# WebSocket endpoint for image editing
@app.websocket("/edit")
async def edit_endpoint(websocket: WebSocket):
    global generated_image_data, pipe_i2i
    await websocket.accept()

    if _generation_in_progress:
        await websocket.send_json({"type": "error", "message": "Another generation is already in progress. Please wait."})
        await websocket.close()
        return

    async with _generation_lock:
        if pipe_i2i is None:
            try:
                await websocket.send_json({"type": "text", "message": "Loading I2I model..."})
                load_i2i_model()
                await websocket.send_json({"type": "text", "message": "I2I Model loaded successfully!"})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                await websocket.close()
                return

        _start_generation()
        websocket_active = True
        start_time = None

        try:
            data = await websocket.receive_json()
            image_base64 = data.get("image_base64", "")
            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "")
            num_inference_steps = data.get("num_inference_steps", 25)
            guidance_scale = data.get("guidance_scale", 7.5)
            strength = data.get("strength", 0.7)
            lora_weight = data.get("lora_weight", 1.0)

            if not image_base64:
                await websocket.send_json({"type": "error", "message": "No image provided"})
                return

            # Decode base64 image
            try:
                image_data = base64.b64decode(image_base64)
                init_image = Image.open(BytesIO(image_data)).convert("RGB")
                logger.info(f"Loaded image for editing: {init_image.size}")
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Failed to decode image: {str(e)}"})
                return

            logger.info(f"Starting I2I editing: steps: {num_inference_steps}, guidance: {guidance_scale}, strength: {strength}")

            # Set LoRA weight
            try:
                if hasattr(pipe_i2i, 'get_list_adapters') and lora_name in pipe_i2i.get_list_adapters():
                    pipe_i2i.set_adapters([lora_name], adapter_weights=[lora_weight])
                    logger.info(f"Set I2I LoRA weight to {lora_weight}")
            except Exception as e:
                logger.warning(f"Could not adjust I2I LoRA weight: {e}")

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
                    logger.error(f"Error in I2I progress callback: {e}")
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
                    logger.error(f"Failed to send I2I progress: {e}")
                    websocket_active = False

            def edit_image():
                _mark_activity()
                try:
                    # Use proper parameters based on pipeline type
                    if pipe_device == "cuda":
                        with torch.cuda.amp.autocast():
                            result = pipe_i2i(
                                prompt=prompt,
                                image=init_image,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                strength=strength,
                                callback_on_step_end=progress_callback,
                                callback_on_step_end_tensor_inputs=["latents"]
                            )
                    else:
                        result = pipe_i2i(
                            prompt=prompt,
                            image=init_image,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            strength=strength,
                            callback_on_step_end=progress_callback,
                            callback_on_step_end_tensor_inputs=["latents"]
                        )
                    
                    raw_image = result.images[0]
                    return _sanitize_image(raw_image)
                except Exception as e:
                    logger.error(f"Error during I2I editing: {e}")
                    raise

            _mark_activity()
            image = await asyncio.get_event_loop().run_in_executor(None, edit_image)
            _mark_activity()

            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG", optimize=True)
            img_byte_arr.seek(0)
            generated_image_data = img_byte_arr.getvalue()

            if websocket_active and getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                await websocket.send_json({"type": "complete"})
                logger.info("I2I editing completed!")

        except WebSocketDisconnect:
            logger.info("I2I WebSocket disconnected by client")
            websocket_active = False
        except Exception as e:
            logger.error(f"I2I Edit error: {e}")
            websocket_active = False
            try:
                if getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({"type": "error", "message": str(e)})
            except:
                pass
        finally:
            _end_generation()
            websocket_active = False
            try:
                if getattr(websocket, "client_state", None) and websocket.client_state.name == "CONNECTED":
                    await websocket.close()
            except:
                pass

# REST endpoints
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

@app.post("/generate-direct")
async def generate_direct(request: ImageGenerationRequest):
    global pipe_t2i, generated_image_data
    
    if _generation_in_progress:
        raise HTTPException(status_code=429, detail="Another generation is already in progress. Please wait.")
    
    async with _generation_lock:
        if pipe_t2i is None:
            try:
                load_t2i_model()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load T2I model: {str(e)}")

        _start_generation()

        try:
            # Set LoRA weight
            try:
                if hasattr(pipe_t2i, 'get_list_adapters') and lora_name in pipe_t2i.get_list_adapters():
                    pipe_t2i.set_adapters([lora_name], adapter_weights=[request.lora_weight])
                    logger.info(f"Set T2I LoRA weight to {request.lora_weight}")
            except Exception as e:
                logger.warning(f"Could not adjust T2I LoRA weight: {e}")

            # Ensure dimensions are divisible by 8
            width = (request.width // 8) * 8
            height = (request.height // 8) * 8

            def generate_image():
                _mark_activity()
                try:
                    if pipe_device == "cuda":
                        with torch.cuda.amp.autocast():
                            result = pipe_t2i(
                                prompt=request.prompt,
                                negative_prompt=request.negative_prompt,
                                num_inference_steps=request.num_inference_steps,
                                guidance_scale=request.guidance_scale,
                                width=width,
                                height=height,
                            )
                    else:
                        result = pipe_t2i(
                            prompt=request.prompt,
                            negative_prompt=request.negative_prompt,
                            num_inference_steps=request.num_inference_steps,
                            guidance_scale=request.guidance_scale,
                            width=width,
                            height=height,
                        )
                    
                    raw_image = result.images[0]
                    return _sanitize_image(raw_image)
                except Exception as e:
                    logger.error(f"Error during direct T2I generation: {e}")
                    raise

            _mark_activity()
            image = await asyncio.get_event_loop().run_in_executor(None, generate_image)
            _mark_activity()

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
            logger.error(f"Direct T2I generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        finally:
            _end_generation()

@app.post("/edit-direct")
async def edit_direct(request: ImageEditRequest):
    global pipe_i2i, generated_image_data
    
    if _generation_in_progress:
        raise HTTPException(status_code=429, detail="Another generation is already in progress. Please wait.")
    
    async with _generation_lock:
        if pipe_i2i is None:
            try:
                load_i2i_model()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load I2I model: {str(e)}")

        _start_generation()

        try:
            # Set LoRA weight
            try:
                if hasattr(pipe_i2i, 'get_list_adapters') and lora_name in pipe_i2i.get_list_adapters():
                    pipe_i2i.set_adapters([lora_name], adapter_weights=[request.lora_weight])
                    logger.info(f"Set I2I LoRA weight to {request.lora_weight}")
            except Exception as e:
                logger.warning(f"Could not adjust I2I LoRA weight: {e}")

            # Decode base64 image
            try:
                image_data = base64.b64decode(request.image_base64)
                init_image = Image.open(BytesIO(image_data)).convert("RGB")
                logger.info(f"Loaded image for editing: {init_image.size}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

            def edit_image():
                _mark_activity()
                try:
                    if pipe_device == "cuda":
                        with torch.cuda.amp.autocast():
                            result = pipe_i2i(
                                prompt=request.prompt,
                                image=init_image,
                                negative_prompt=request.negative_prompt,
                                num_inference_steps=request.num_inference_steps,
                                guidance_scale=request.guidance_scale,
                                strength=request.strength,
                            )
                    else:
                        result = pipe_i2i(
                            prompt=request.prompt,
                            image=init_image,
                            negative_prompt=request.negative_prompt,
                            num_inference_steps=request.num_inference_steps,
                            guidance_scale=request.guidance_scale,
                            strength=request.strength,
                        )
                    
                    raw_image = result.images[0]
                    return _sanitize_image(raw_image)
                except Exception as e:
                    logger.error(f"Error during direct I2I edit: {e}")
                    raise

            _mark_activity()
            image = await asyncio.get_event_loop().run_in_executor(None, edit_image)
            _mark_activity()

            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG", optimize=True)
            img_byte_arr.seek(0)
            generated_image_data = img_byte_arr.getvalue()
            _mark_activity()

            return StreamingResponse(
                BytesIO(generated_image_data),
                media_type="image/png",
                headers={"Content-Disposition": "attachment; filename=edited_image.png"}
            )
        except Exception as e:
            logger.error(f"Direct I2I edit error: {e}")
            raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")
        finally:
            _end_generation()

# Status endpoints
@app.get("/")
async def root():
    global pipe_t2i, pipe_i2i, pipe_device, pipe_dtype, model_path, lora_path, lora_file
    t2i_loaded = pipe_t2i is not None
    i2i_loaded = pipe_i2i is not None
    adapters = []
    if t2i_loaded and hasattr(pipe_t2i, 'get_list_adapters'):
        try:
            adapters = pipe_t2i.get_list_adapters()
        except:
            pass
    elif i2i_loaded and hasattr(pipe_i2i, 'get_list_adapters'):
        try:
            adapters = pipe_i2i.get_list_adapters()
        except:
            pass
    
    return {
        "message": "Text-to-Image and Image Edit API is running",
        "model_path": model_path,
        "model": os.path.basename(model_path) if model_path else "No model path set",
        "t2i_model_loaded": t2i_loaded,
        "i2i_model_loaded": i2i_loaded,
        "lora_path": lora_path,
        "lora_file": lora_file,
        "lora_loaded": lora_name in adapters,
        "device": pipe_device,
        "dtype": str(pipe_dtype) if pipe_dtype else None,
        "generation_in_progress": _generation_in_progress,
        "endpoints": {
            "text_to_image": ["/generate", "/progress", "/generate-direct"],
            "image_edit": ["/edit", "/edit-direct"],
            "get_image": "/generate-image"
        }
    }

@app.get("/model-info")
async def model_info():
    global pipe_t2i, pipe_i2i, pipe_device, pipe_dtype, model_path, lora_path, lora_file
    adapters = []
    if pipe_t2i and hasattr(pipe_t2i, 'get_list_adapters'):
        try:
            adapters = pipe_t2i.get_list_adapters()
        except:
            pass
    elif pipe_i2i and hasattr(pipe_i2i, 'get_list_adapters'):
        try:
            adapters = pipe_i2i.get_list_adapters()
        except:
            pass
    
    return {
        "model_name": os.path.basename(model_path) if model_path else "No model path set",
        "model_path": model_path,
        "lora_path": lora_path,
        "lora_file": lora_file,
        "t2i_model_loaded": pipe_t2i is not None,
        "i2i_model_loaded": pipe_i2i is not None,
        "device": pipe_device,
        "dtype": str(pipe_dtype) if pipe_dtype else "unknown",
        "loaded_adapters": adapters,
        "capabilities": ["text-to-image", "image-to-image"],
        "generation_in_progress": _generation_in_progress
    }

@app.post("/load-t2i")
async def load_t2i_endpoint():
    try:
        load_t2i_model()
        return {"message": "T2I model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load T2I model: {str(e)}")

@app.post("/load-i2i")
async def load_i2i_endpoint():
    try:
        load_i2i_model()
        return {"message": "I2I model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load I2I model: {str(e)}")

@app.post("/unload-models")
async def unload_models_endpoint():
    success = await _async_unload_models()
    return {"message": "Models unloaded" if success else "Models were not loaded"}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Text-to-Image and Image Edit API Server with LoRA support")
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
        default="default",
        help="Name for the LoRA adapter (default: default)"
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
        default=8281,
        help="Port to bind the server to (default: 8281)"
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
    
    logger.info(f"Starting Text-to-Image and Image Edit API server")
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