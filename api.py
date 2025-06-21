import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image
import io
import torch
import random
import time
import os
import sys
import numpy as np
import cv2 # For RealESRGAN
import traceback

# --- Real-ESRGAN Upscaler Imports ---
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE_API = True
except ImportError:
    print("API Warning: RealESRGAN dependencies not found. Upscaling will be disabled in API.")
    REALESRGAN_AVAILABLE_API = False
    RealESRGANer = None
    RRDBNet = None

# --- Configuration & Global State ---
# Attempt to import from app.py, otherwise define locally
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app import (
        STYLE_PRESETS as APP_STYLE_PRESETS,
        apply_style as app_apply_style,
        parse_dimensions as app_parse_dimensions,
        PREFERRED_SDXL_MODEL_ID as APP_PREFERRED_SDXL_MODEL_ID,
        FALLBACK_SD15_MODEL_ID as APP_FALLBACK_SD15_MODEL_ID
    )
    STYLE_PRESETS = APP_STYLE_PRESETS
    apply_style = app_apply_style
    parse_dimensions = app_parse_dimensions
    # API will use its own model ID variables, but can default to app's
    API_PREFERRED_SDXL_MODEL_ID = os.getenv("API_SDXL_MODEL_ID", APP_PREFERRED_SDXL_MODEL_ID)
    API_FALLBACK_SD15_MODEL_ID = os.getenv("API_SD15_MODEL_ID", APP_FALLBACK_SD15_MODEL_ID)
    print("API: Successfully imported some shared components from app.py.")
except ImportError as e:
    print(f"API Warning: Could not import from app.py ({e}). Redefining components for API.")
    API_PREFERRED_SDXL_MODEL_ID = os.getenv("API_SDXL_MODEL_ID", "stabilityai/sdxl-base-1.0")
    API_FALLBACK_SD15_MODEL_ID = os.getenv("API_SD15_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    STYLE_PRESETS = {
        "None": {"prompt_suffix": "", "negative_prompt_prefix": ""},
        "Realistic": {"prompt_suffix": "photorealistic, 4k, ultra detailed", "negative_prompt_prefix": "cartoon, anime"},
        "Cyberpunk": {"prompt_suffix": "cyberpunk, neon lights, futuristic", "negative_prompt_prefix": "historical, nature"},
        "Anime": {"prompt_suffix": "anime style, key visual, beautiful", "negative_prompt_prefix": "photorealistic, 3d"},
        "Watercolor": {"prompt_suffix": "watercolor painting, soft wash", "negative_prompt_prefix": "photorealistic, harsh lines"},
        "3D Render": {"prompt_suffix": "3d render, octane render, detailed", "negative_prompt_prefix": "2d, painting"},
    }
    def apply_style(prompt, style_name):
        if style_name == "None" or style_name not in STYLE_PRESETS: return prompt, ""
        preset = STYLE_PRESETS[style_name]
        styled_p = f"{prompt.strip()}, {preset['prompt_suffix']}" if prompt.strip() else preset['prompt_suffix']
        return styled_p.strip(", "), preset['negative_prompt_prefix'].strip(", ")
    def parse_dimensions(dim_string):
        try: w, h = map(int, dim_string.split('x')); return w, h
        except: return 1024, 1024 # Default for API if parsing fails (SDXL focus)

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoPipelineForText2Image # Always needed

# API's own model instances
api_pipe = None
api_current_model_id = None
api_loaded_model_type = None # 'sdxl' or 'sd1.5'
api_upscaler_instance = None
REALESRGAN_API_MODEL_NAME = 'RealESRGAN_x4plus'
REALESRGAN_API_SCALE = 4


# --- API Model Loading Logic ---
def load_api_diffusion_pipeline(model_id, is_sdxl, use_float16, use_attention_slicing, status_fn):
    global api_pipe # Modifies API's global pipe

    args = {"torch_dtype": torch.float16} if torch.cuda.is_available() and use_float16 else {}
    status_fn(f"API: Loading {'SDXL' if is_sdxl else 'non-SDXL'} model: {model_id} with float16={use_float16}")

    if is_sdxl:
        api_pipe = AutoPipelineForText2Image.from_pretrained(model_id, **args)
    else:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        args["scheduler"] = scheduler
        api_pipe = StableDiffusionPipeline.from_pretrained(model_id, **args)

    if torch.cuda.is_available(): api_pipe.to("cuda")
    if use_attention_slicing and hasattr(api_pipe, "enable_attention_slicing"):
        api_pipe.enable_attention_slicing()
        status_fn(f"API: Attention slicing enabled for {model_id}.")
    return model_id, "sdxl" if is_sdxl else "sd1.5"

def load_api_diffusion_model_with_fallback(preferred_id, fallback_id, use_fp16, use_attn_slicing, status_fn):
    global api_pipe, api_current_model_id, api_loaded_model_type

    is_preferred_sdxl = "sdxl" in preferred_id.lower()
    try:
        api_current_model_id, api_loaded_model_type = load_api_diffusion_pipeline(preferred_id, is_preferred_sdxl, use_fp16, use_attn_slicing, status_fn)
        status_fn(f"API: Successfully loaded preferred model: {api_current_model_id} ({api_loaded_model_type})")
        return True
    except Exception as e_pref:
        status_fn(f"API: Failed to load preferred model '{preferred_id}': {e_pref}. Trying fallback.")
        traceback.print_exc()
        api_pipe = None
        if fallback_id:
            try:
                api_current_model_id, api_loaded_model_type = load_api_diffusion_pipeline(fallback_id, False, use_fp16, use_attn_slicing, status_fn) # Fallback assumed non-SDXL if not specified
                status_fn(f"API: Successfully loaded fallback model: {api_current_model_id} ({api_loaded_model_type})")
                return True
            except Exception as e_fall:
                status_fn(f"API: Failed to load fallback model '{fallback_id}': {e_fall}")
                traceback.print_exc()
    api_current_model_id, api_loaded_model_type = None, None
    return False

def load_api_upscaler(status_fn):
    global api_upscaler_instance
    if not REALESRGAN_AVAILABLE_API:
        status_fn("API: RealESRGAN unavailable, upscaling disabled.")
        return False
    if api_upscaler_instance: return True

    status_fn(f"API: Loading RealESRGAN: {REALESRGAN_API_MODEL_NAME}...")
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=REALESRGAN_API_SCALE)
        half = torch.cuda.is_available()
        api_upscaler_instance = RealESRGANer(
            scale=REALESRGAN_API_SCALE, model_path=None, model=model, dni_weight=None,
            model_name=REALESRGAN_API_MODEL_NAME, tile=0, tile_pad=10, pre_pad=0,
            half=half, gpu_id=0 if half else None
        )
        status_fn("API: RealESRGAN upscaler loaded.")
        return True
    except Exception as e:
        status_fn(f"API: Error loading RealESRGAN: {e}")
        traceback.print_exc()
        api_upscaler_instance = None
        return False

def upscale_api_image(pil_img):
    if not api_upscaler_instance: return pil_img
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    try:
        output_cv, _ = api_upscaler_instance.enhance(cv_img, outscale=REALESRGAN_API_SCALE)
        return Image.fromarray(cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"API: Upscaling error: {e}"); traceback.print_exc()
        return pil_img


# --- FastAPI App ---
app = FastAPI(title="Ultra Professional Image Generation API", version="1.0.0")
api_models_loaded_successfully = False

@app.on_event("startup")
async def startup_api_models():
    global api_models_loaded_successfully
    print_status = lambda msg: print(f"[API Startup] {msg}") # Simple console logger for startup

    use_fp16 = os.getenv("API_USE_FLOAT16", "true").lower() == "true"
    use_attn_slicing = os.getenv("API_ATTENTION_SLICING", "true").lower() == "true"

    diffusion_loaded = load_api_diffusion_model_with_fallback(
        API_PREFERRED_SDXL_MODEL_ID, API_FALLBACK_SD15_MODEL_ID,
        use_fp16, use_attn_slicing, print_status
    )
    upscaler_loaded = False
    if REALESRGAN_AVAILABLE_API:
        upscaler_loaded = load_api_upscaler(print_status)

    api_models_loaded_successfully = diffusion_loaded # At least diffusion model must load
    if api_models_loaded_successfully:
        print_status(f"Diffusion model '{api_current_model_id}' ready.")
        if upscaler_loaded: print_status("Upscaler ready.")
        else: print_status("Upscaler not loaded/available.")
    else:
        print_status("CRITICAL: No diffusion model could be loaded. API will be impaired.")

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, example="A hyperrealistic cat astronaut on the moon, detailed fur, cosmic background")
    negative_prompt: str = Field("", example="blurry, low quality, text, watermark, human")
    style_name: str = Field("Realistic", example="Realistic")
    dimensions_str: str = Field("1024x1024", example="1024x1024", description="WxH format, e.g., 512x512, 1024x1024")
    num_inference_steps: int = Field(25, ge=10, le=50)
    guidance_scale: float = Field(7.0, ge=1.0, le=15.0)
    seed: int = Field(-1, description="Seed for reproducibility. -1 for random.")
    upscale_active: bool = Field(True, description="Enable Real-ESRGAN x4 upscaling.")

@app.post("/generate/", response_class=StreamingResponse)
async def generate_image_endpoint(request: GenerationRequest):
    if not api_models_loaded_successfully or not api_pipe:
        raise HTTPException(status_code=503, detail="Diffusion model not available.")

    if request.style_name not in STYLE_PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid style: {request.style_name}. Available: {list(STYLE_PRESETS.keys())}")

    width, height = parse_dimensions(request.dimensions_str)

    # Basic dimension check for API (could be more sophisticated)
    if api_loaded_model_type == "sd1.5" and (width > 768 or height > 768):
         print(f"API Warning: SD1.5 requested with {width}x{height}. May be unstable. Max 768px recommended for SD1.5.")
    elif api_loaded_model_type == "sdxl" and (width < 768 or height < 768):
         print(f"API Warning: SDXL requested with {width}x{height}. Suboptimal results likely below 768px (1024px ideal).")


    styled_prompt, style_neg = apply_style(request.prompt, request.style_name)
    final_neg_prompt = request.negative_prompt
    if style_neg: final_neg_prompt = f"{style_neg}, {final_neg_prompt}" if final_neg_prompt else style_neg

    actual_seed = request.seed if request.seed != -1 else random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)

    params = {"prompt": styled_prompt, "negative_prompt": final_neg_prompt or None,
              "num_inference_steps": request.num_inference_steps, "guidance_scale": request.guidance_scale,
              "generator": generator, "width": width, "height": height}
    try:
        start_time = time.time()
        print(f"API: Generating {width}x{height} image for prompt: '{styled_prompt[:50]}...' Seed: {actual_seed}")
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            img = api_pipe(**params).images[0]

        logs = [f"Initial image {img.width}x{img.height} generated."]

        if request.upscale_active:
            if REALESRGAN_AVAILABLE_API and api_upscaler_instance:
                print("API: Upscaling image...")
                img = upscale_api_image(img)
                logs.append(f"Image upscaled to {img.width}x{img.height}.")
            elif REALESRGAN_AVAILABLE_API and not api_upscaler_instance: # Attempt to load if missing
                print("API: Upscaler not loaded, trying to load now for this request...")
                if load_api_upscaler(print) and api_upscaler_instance:
                     img = upscale_api_image(img); logs.append(f"Image upscaled to {img.width}x{img.height} (adhoc load).")
                else: logs.append("API: Upscaling skipped, upscaler failed to load adhoc.")
            else: logs.append("API: Upscaling skipped, RealESRGAN not available.")
        else: logs.append("API: Upscaling disabled by request.")

        end_time = time.time()
        print(f"API: Total processing time: {end_time - start_time:.2f}s. Final size: {img.width}x{img.height}")

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Include some generation info in headers (optional)
        headers = {
            "X-Seed-Used": str(actual_seed),
            "X-Model-Used": str(api_current_model_id),
            "X-Final-Dimensions": f"{img.width}x{img.height}",
            "X-Upscaled": str(request.upscale_active and img.width > width), # True if upscaling was active AND dimensions changed
            "X-Generation-Logs": "; ".join(logs) # Simple way to pass some logs
        }
        return StreamingResponse(img_byte_arr, media_type="image/png", headers=headers)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health/")
async def health():
    return {
        "status": "ok" if api_models_loaded_successfully and api_pipe else "error_model_load",
        "diffusion_model_id": api_current_model_id,
        "diffusion_model_type": api_loaded_model_type,
        "upscaler_model_name": REALESRGAN_API_MODEL_NAME if api_upscaler_instance else None,
        "upscaler_available": REALESRGAN_AVAILABLE_API,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("API_LOG_LEVEL", "info")

    print(f"Starting API server on {host}:{port}. Reload: {reload}. Log Level: {log_level}")
    print(f"Default Diffusion Model (env API_SDXL_MODEL_ID): {API_PREFERRED_SDXL_MODEL_ID}")
    print(f"Fallback Diffusion Model (env API_SD15_MODEL_ID): {API_FALLBACK_SD15_MODEL_ID}")
    print(f"Upscaler Model: {REALESRGAN_API_MODEL_NAME if REALESRGAN_AVAILABLE_API else 'Not Available'}")

    uvicorn.run("api:app", host=host, port=port, reload=reload, log_level=log_level)
