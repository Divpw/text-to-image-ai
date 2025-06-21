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

# Attempt to import necessary components from app.py or define them here
# This assumes app.py has these defined at the top level or in a way that's importable.
# If app.py is primarily a script, we might need to duplicate or refactor them into a shared utils.py
try:
    from app import STYLE_PRESETS, apply_style, load_model as load_diffusion_model, current_model_id as app_current_model_id, pipe as app_pipe, DEFAULT_MODEL_ID
except ImportError:
    print("Warning: Could not import directly from app.py. Redefining necessary components for API.")
    # Redefine if import fails (e.g., if app.py is not in PYTHONPATH or has script-like execution guards)
    DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    STYLE_PRESETS = {
        "None": {"prompt_suffix": "", "negative_prompt_prefix": ""},
        "Realistic": {"prompt_suffix": "photorealistic, 4k, ultra detailed, cinematic lighting", "negative_prompt_prefix": "cartoon, anime, drawing, sketch, stylized"},
        "Anime": {"prompt_suffix": "anime style, key visual, vibrant, beautiful, detailed", "negative_prompt_prefix": "photorealistic, 3d render"},
        # Add more styles if needed, keep consistent with app.py
    }
    def apply_style(prompt, style_name):
        if style_name == "None" or style_name not in STYLE_PRESETS: return prompt, ""
        preset = STYLE_PRESETS[style_name]; return f"{prompt}, {preset['prompt_suffix']}", preset['negative_prompt_prefix']

    # Simplified model loader for API context (can be expanded)
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoPipelineForText2Image
    app_pipe = None
    app_current_model_id = None

    def load_diffusion_model(model_id_to_load=DEFAULT_MODEL_ID, use_float16=True, use_attention_slicing=False, status_update_fn=None):
        global app_pipe, app_current_model_id
        if app_pipe is not None and app_current_model_id == model_id_to_load:
            if status_update_fn: status_update_fn(f"API: Model '{model_id_to_load}' already loaded.")
            return True # Indicate loaded

        if status_update_fn: status_update_fn(f"API: Loading model: {model_id_to_load}...")
        pipeline_args = {}
        if torch.cuda.is_available() and use_float16: pipeline_args["torch_dtype"] = torch.float16

        model_is_sdxl = "sdxl" in model_id_to_load.lower()
        try:
            if model_is_sdxl:
                app_pipe = AutoPipelineForText2Image.from_pretrained(model_id_to_load, **pipeline_args)
            else:
                scheduler = EulerDiscreteScheduler.from_pretrained(model_id_to_load, subfolder="scheduler")
                pipeline_args["scheduler"] = scheduler
                app_pipe = StableDiffusionPipeline.from_pretrained(model_id_to_load, **pipeline_args)

            if torch.cuda.is_available(): app_pipe.to("cuda")
            if use_attention_slicing and hasattr(app_pipe, "enable_attention_slicing"): app_pipe.enable_attention_slicing()

            app_current_model_id = model_id_to_load
            if status_update_fn: status_update_fn(f"API: Model '{model_id_to_load}' loaded successfully.")
            return True
        except Exception as e:
            if status_update_fn: status_update_fn(f"API: Error loading model '{model_id_to_load}': {e}")
            app_pipe = None; app_current_model_id = None; return False


# --- FastAPI App Setup ---
app = FastAPI(title="Stable Diffusion Image Generation API", version="0.1.0")

# --- Pydantic Models for Request Body ---
class GenerationRequest(BaseModel):
    prompt: str = Field(..., example="A hyperrealistic cat astronaut on the moon")
    negative_prompt: str = Field("", example="blurry, low quality, text, watermark")
    style_name: str = Field("None", example="Realistic")
    num_inference_steps: int = Field(25, ge=10, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    seed: int = Field(None, description="Optional seed for reproducibility. If None, random seed is used.")
    # model_id: str = Field(DEFAULT_MODEL_ID, description="Specify model if different from loaded one. Requires model reloading.") # Future: allow model change

# --- API State ---
api_model_loaded = False
API_MODEL_ID = os.getenv("API_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID) # Use environment variable or default

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    global api_model_loaded, API_MODEL_ID
    print("API Startup: Attempting to load model...")
    # Determine if float16 and attention slicing should be used (can be from env vars or defaults)
    use_fp16_env = os.getenv("API_USE_FLOAT16", "true").lower() == "true"
    use_attn_slicing_env = os.getenv("API_ATTENTION_SLICING", "false").lower() == "true"

    api_model_loaded = load_diffusion_model(
        model_id_to_load=API_MODEL_ID,
        use_float16=use_fp16_env,
        use_attention_slicing=use_attn_slicing_env,
        status_update_fn=print # Print status to console
    )
    if not api_model_loaded:
        print(f"API Startup: CRITICAL - Model '{API_MODEL_ID}' failed to load. API may not function.")
    else:
        print(f"API Startup: Model '{API_MODEL_ID}' loaded. API is ready.")

# --- API Endpoints ---
@app.post("/generate/", response_class=StreamingResponse)
async def generate_image_api(request: GenerationRequest):
    global app_pipe, app_current_model_id # Use the (potentially imported) pipe and model_id

    if not api_model_loaded or app_pipe is None:
        raise HTTPException(status_code=503, detail=f"Model '{API_MODEL_ID}' is not loaded or failed to load. API unavailable.")

    print(f"API Request: Prompt='{request.prompt}', Style='{request.style_name}', Seed='{request.seed}'")

    styled_prompt, style_negative_prefix = apply_style(request.prompt, request.style_name)
    if style_negative_prefix and request.negative_prompt:
        final_negative_prompt = f"{style_negative_prefix}, {request.negative_prompt}"
    elif style_negative_prefix:
        final_negative_prompt = style_negative_prefix
    else:
        final_negative_prompt = request.negative_prompt

    current_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(current_seed)

    generation_args = {
        "prompt": styled_prompt,
        "negative_prompt": final_negative_prompt if final_negative_prompt else None,
        "num_inference_steps": int(request.num_inference_steps),
        "guidance_scale": float(request.guidance_scale),
        "generator": generator
    }

    try:
        print(f"API: Generating image with model {app_current_model_id}...")
        start_time = time.time()

        # Ensure using the correct data type for the loaded pipe
        if device == "cuda" and hasattr(app_pipe, 'torch_dtype') and app_pipe.torch_dtype == torch.float16:
            with torch.autocast("cuda"):
                pil_image = app_pipe(**generation_args).images[0]
        else:
            pil_image = app_pipe(**generation_args).images[0]

        end_time = time.time()
        print(f"API: Image generated in {end_time - start_time:.2f} seconds. Seed used: {current_seed}")

        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0) # Reset stream position

        # Suggest a filename (optional, client might ignore)
        # sane_prompt_prefix = "".join(c if c.isalnum() else "_" for c in styled_prompt[:20])
        # suggested_filename = f"{sane_prompt_prefix}_{app_current_model_id.split('/')[-1]}_{current_seed}_{time.strftime('%Y%m%d%H%M%S')}.png"

        return StreamingResponse(img_byte_arr, media_type="image/png") #, headers={"Content-Disposition": f"inline; filename=\"{suggested_filename}\""})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"API: Error during image generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during image generation: {str(e)}")

@app.get("/health/")
async def health_check():
    model_status = "loaded" if api_model_loaded and app_pipe is not None else "not loaded or error"
    return {
        "status": "ok",
        "model_id": app_current_model_id or API_MODEL_ID,
        "model_status": model_status,
        "cuda_available": torch.cuda.is_available()
    }

# --- Main Execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    # Configuration for Uvicorn can be set here or passed via CLI
    # Example: uvicorn api:app --reload --port 8000
    print("Starting API with Uvicorn...")
    print(f"Default model for API (can be overridden by env var API_DEFAULT_MODEL_ID): {API_MODEL_ID}")
    print("To run: uvicorn api:app --reload --host 0.0.0.0 --port 8000")

    # This part is mostly for information; `uvicorn api:app` is the typical way to run.
    # uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") # This would block if uncommented
    # The startup event will handle model loading when Uvicorn starts the app.
```

**Explanation of `api.py`:**

1.  **Imports**: Includes FastAPI, Uvicorn, Pydantic for request models, PIL for image handling, and basic Python libraries.
2.  **Shared Components Import/Redefinition**:
    *   It *tries* to import `STYLE_PRESETS`, `apply_style`, `load_model` etc., from `app.py`.
    *   If the import fails (e.g., `app.py` is run as a script and not easily importable as a module, or path issues), it redefines these essential components. This makes `api.py` more robust for standalone execution.
    *   The model `pipe` and `current_model_id` are aliased as `app_pipe` and `app_current_model_id` to avoid name clashes if imported.
3.  **FastAPI App Initialization**: `app = FastAPI(...)`.
4.  **Pydantic Model (`GenerationRequest`)**: Defines the expected structure and types for the POST request body, including default values and validation (e.g., `ge`, `le` for ranges).
5.  **API State**: `api_model_loaded` flag and `API_MODEL_ID` (configurable via environment variable `API_DEFAULT_MODEL_ID`).
6.  **Startup Event (`@app.on_event("startup")`)**:
    *   This function runs when the FastAPI application starts.
    *   It calls `load_diffusion_model` to load the Stable Diffusion model into memory. This ensures the model is ready before any requests come in.
    *   Configuration for float16 and attention slicing can also be set via environment variables (`API_USE_FLOAT16`, `API_ATTENTION_SLICING`).
7.  **`/generate/` Endpoint (`@app.post("/generate/")`)**:
    *   Accepts a `GenerationRequest`.
    *   Checks if the model is loaded; if not, returns a 503 error.
    *   Applies styles, sets up the generator with the seed.
    *   Calls `app_pipe(**generation_args).images[0]` to generate the image.
    *   Converts the PIL image to PNG bytes.
    *   Returns a `StreamingResponse` with `media_type="image/png"`, which sends the image directly to the client.
    *   Includes error handling for the generation process.
8.  **`/health/` Endpoint (`@app.get("/health/")`)**: A simple health check endpoint to verify the API is running and if the model is loaded.
9.  **`if __name__ == "__main__":`**: Provides information on how to run the API using Uvicorn (e.g., `uvicorn api:app --reload --host 0.0.0.0 --port 8000`).

Next, I'll add instructions to `README.md` for running this API.
