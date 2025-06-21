import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoPipelineForText2Image
from PIL import Image
import random
import argparse
import time
import os
import sys
import numpy as np
import cv2 # For RealESRGAN image handling
import traceback

# --- Real-ESRGAN Upscaler Imports ---
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    print("Warning: RealESRGAN dependencies not found. Upscaling will be disabled.")
    print("To enable upscaling, please install realesrgan and basicsr: pip install realesrgan basicsr opencv-python")
    REALESRGAN_AVAILABLE = False
    RealESRGANer = None
    RRDBNet = None


# --- Configuration & Global State ---
# Main Diffusion Model
pipe = None
current_model_id = None
loaded_model_type = None # 'sdxl', 'sd1.5', or None

# Default to SD1.5 for CPU compatibility on Hugging Face Spaces free tier
DEFAULT_MODEL_FOR_CPU_TIER = "runwayml/stable-diffusion-v1-5"
# SDXL model ID, can be specified via CLI if a GPU is available
SDXL_MODEL_ID_OPTION = "stabilityai/sdxl-base-1.0"
# Fallback is essentially the default for CPU if SDXL is attempted and fails or not specified
FALLBACK_SD15_MODEL_ID = DEFAULT_MODEL_FOR_CPU_TIER

# Upscaler Model
upscaler_instance = None
REALESRGAN_MODEL_NAME = 'RealESRGAN_x4plus' # General purpose x4 upscaler
REALESRGAN_SCALE = 4

# --- Style Presets ---
STYLE_PRESETS = {
    "None": {"prompt_suffix": "", "negative_prompt_prefix": ""},
    "Realistic": {"prompt_suffix": "photorealistic, 4k, ultra detailed, cinematic lighting, professional photography", "negative_prompt_prefix": "cartoon, anime, drawing, sketch, stylized, illustration, painting, art, text, watermark, signature, ugly, deformed"},
    "Cyberpunk": {"prompt_suffix": "cyberpunk cityscape, neon lights, futuristic, dystopian, highly detailed, intricate", "negative_prompt_prefix": "historical, medieval, nature, daytime, sunny, ugly, deformed"},
    "Anime": {"prompt_suffix": "anime style, key visual, vibrant, beautiful, detailed illustration, official art, studio ghibli, makoto shinkai", "negative_prompt_prefix": "photorealistic, 3d render, ugly, disfigured, real life, text, watermark"},
    "Watercolor": {"prompt_suffix": "watercolor painting, soft wash, wet-on-wet, flowing colors, detailed brushstrokes", "negative_prompt_prefix": "photorealistic, harsh lines, 3d render, octane render, text, signature"},
    "3D Render": {"prompt_suffix": "3d render, octane render, blender, vray, detailed textures, physically based rendering, cinematic", "negative_prompt_prefix": "2d, drawing, sketch, painting, illustration, flat, cartoon, text"},
}

# --- Helper Functions ---
def get_random_seed():
    return random.randint(0, 2**32 - 1)

def parse_dimensions(dim_string):
    try:
        w, h = map(int, dim_string.split('x'))
        return w, h
    except:
        print(f"Warning: Could not parse dimensions '{dim_string}'. Defaulting to 512x512.")
        return 512, 512

def apply_style(prompt, style_name):
    if style_name == "None" or style_name not in STYLE_PRESETS:
        return prompt, ""
    preset = STYLE_PRESETS[style_name]
    styled_prompt = f"{prompt.strip()}, {preset['prompt_suffix']}" if prompt.strip() else preset['prompt_suffix']
    return styled_prompt.strip(", "), preset['negative_prompt_prefix'].strip(", ")


# --- Model Loading ---
def load_diffusion_pipeline_internal(model_id_to_load, is_sdxl_attempt, use_float16_config, use_attention_slicing_config, status_update_fn):
    """Internal logic to load a specific diffusion pipeline."""
    global pipe # Modifies the global pipe instance

    actual_use_float16 = torch.cuda.is_available() and use_float16_config

    pipeline_args = {}
    if actual_use_float16:
        pipeline_args["torch_dtype"] = torch.float16
        status_update_fn(f"Using torch_dtype: float16 for {model_id_to_load}")
    else:
        status_update_fn(f"Using default torch_dtype (float32) for {model_id_to_load} (CUDA available: {torch.cuda.is_available()}, float16_config: {use_float16_config})")


    if is_sdxl_attempt:
        status_update_fn(f"Attempting to load SDXL model: {model_id_to_load}...")
        if not torch.cuda.is_available():
            status_update_fn("CRITICAL WARNING: Attempting to load SDXL on CPU. This will be extremely slow and likely cause memory issues. Not recommended for SDXL.")
        pipe = AutoPipelineForText2Image.from_pretrained(model_id_to_load, **pipeline_args)
    else: # SD 1.5 or other non-SDXL
        status_update_fn(f"Attempting to load non-SDXL model: {model_id_to_load}...")
        # For non-SDXL, ensure scheduler is correctly passed if needed by StableDiffusionPipeline
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id_to_load, subfolder="scheduler")
        pipeline_args["scheduler"] = scheduler
        pipe = StableDiffusionPipeline.from_pretrained(model_id_to_load, **pipeline_args)

    if torch.cuda.is_available():
        status_update_fn(f"Moving {model_id_to_load} to CUDA...")
        pipe.to("cuda")
    else:
        status_update_fn(f"CUDA not available. {model_id_to_load} will run on CPU (expect very slow performance).")

    actual_use_attention_slicing = torch.cuda.is_available() and use_attention_slicing_config
    if actual_use_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        status_update_fn(f"Enabling attention slicing for {model_id_to_load}.")
        pipe.enable_attention_slicing()

    status_update_fn(f"Model {model_id_to_load} loaded successfully.")
    return model_id_to_load, "sdxl" if is_sdxl_attempt else "sd1.5"


def load_model(requested_model_id, use_float16_config, use_attention_slicing_config, status_update_fn=None):
    """
    Loads the diffusion model.
    Prioritizes requested_model_id. If it's SDXL and fails on a system that might support SD1.5,
    it attempts to fall back to FALLBACK_SD15_MODEL_ID.
    """
    global pipe, current_model_id, loaded_model_type

    if status_update_fn is None: status_update_fn = print

    if pipe is not None and current_model_id == requested_model_id:
        status_update_fn(f"Model '{requested_model_id}' is already loaded.")
        return True

    is_requested_sdxl = "sdxl" in requested_model_id.lower()

    try:
        # Attempt to load the requested model
        loaded_id, type_loaded = load_diffusion_pipeline_internal(
            requested_model_id, is_requested_sdxl,
            use_float16_config, use_attention_slicing_config,
            status_update_fn
        )
        current_model_id = loaded_id
        loaded_model_type = type_loaded
        return True
    except Exception as e_requested:
        status_update_fn(f"Failed to load requested model '{requested_model_id}': {e_requested}")
        traceback.print_exc()
        pipe = None

        # Fallback logic: If the requested model was SDXL and it failed,
        # AND the requested model is different from the hardcoded SD1.5 fallback,
        # then try loading the SD1.5 fallback. This avoids trying to load SD1.5 if it was already the one that failed.
        if is_requested_sdxl and requested_model_id != FALLBACK_SD15_MODEL_ID:
            status_update_fn(f"Attempting to load fallback SD 1.5 model: {FALLBACK_SD15_MODEL_ID}...")
            try:
                loaded_id, type_loaded = load_diffusion_pipeline_internal(
                    FALLBACK_SD15_MODEL_ID, False, # is_sdxl_attempt = False
                    use_float16_config, use_attention_slicing_config,
                    status_update_fn
                )
                current_model_id = loaded_id
                loaded_model_type = type_loaded
                return True
            except Exception as e_fallback:
                status_update_fn(f"Failed to load fallback SD 1.5 model '{FALLBACK_SD15_MODEL_ID}': {e_fallback}")
                traceback.print_exc()
                current_model_id = None
                loaded_model_type = None
                return False
        else: # Requested model was not SDXL, or it was the fallback itself that failed
            current_model_id = None
            loaded_model_type = None
            return False

# --- Upscaler Functions ---
def load_upscaler(status_update_fn=None):
    global upscaler_instance
    if status_update_fn is None: status_update_fn = print

    if not REALESRGAN_AVAILABLE:
        status_update_fn("RealESRGAN library not available. Upscaling disabled.")
        return False

    if upscaler_instance is not None:
        status_update_fn("Upscaler model already loaded.")
        return True

    status_update_fn(f"Loading Real-ESRGAN upscaler: {REALESRGAN_MODEL_NAME}...")
    try:
        model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=REALESRGAN_SCALE)

        # RealESRGANer uses half precision if torch.cuda.is_available() and half=True
        # For CPU, half precision is not typically used/beneficial for RealESRGAN.
        use_half_for_upscaler = torch.cuda.is_available()

        upscaler_instance = RealESRGANer(
            scale=REALESRGAN_SCALE, model_path=None, model=model_arch, dni_weight=None,
            model_name=REALESRGAN_MODEL_NAME, tile=0, tile_pad=10, pre_pad=0,
            half=use_half_for_upscaler,
            gpu_id=0 if torch.cuda.is_available() else None
        )
        status_update_fn(f"Real-ESRGAN '{REALESRGAN_MODEL_NAME}' loaded.")
        return True
    except Exception as e:
        status_update_fn(f"Error loading Real-ESRGAN: {e}. Upscaling disabled.")
        traceback.print_exc()
        upscaler_instance = None
        return False

def upscale_image(pil_image, progress_callback=None):
    global upscaler_instance
    if not REALESRGAN_AVAILABLE or upscaler_instance is None:
        print("Upscaling skipped: RealESRGAN not available or model not loaded.")
        return pil_image

    if progress_callback: progress_callback(0.0, desc="✨ Upscaling (Real-ESRGAN)...")

    # Warn if upscaling on CPU
    if not torch.cuda.is_available():
        print("WARNING: Performing Real-ESRGAN upscaling on CPU. This will be very slow.")
        if progress_callback: progress_callback(0.05, desc="✨ Upscaling (CPU - very slow)...")

    print(f"Upscaling image (original size: {pil_image.width}x{pil_image.height}) with {REALESRGAN_MODEL_NAME}...")

    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    try:
        output_cv, _ = upscaler_instance.enhance(cv_image, outscale=REALESRGAN_SCALE)
        output_pil = Image.fromarray(cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB))
        print(f"Image upscaled to: {output_pil.width}x{output_pil.height}")
        if progress_callback: progress_callback(1.0, desc="✨ Upscaling Complete!")
        return output_pil
    except Exception as e:
        print(f"Error during upscaling: {e}")
        traceback.print_exc()
        if progress_callback: progress_callback(1.0, desc="❌ Upscaling Failed.")
        return pil_image


# --- Image Generation ---
def generate_image_fn(prompt, negative_prompt, style_name, dimensions_str,
                        num_inference_steps, guidance_scale, seed_value,
                        custom_filename_prefix, upscale_active, progress=gr.Progress(track_ τότε=True)):
    global pipe, current_model_id, loaded_model_type, upscaler_instance

    if pipe is None:
        return None, "Diffusion model not loaded. Please check logs or restart.", None

    logs = []
    progress(0, desc="🎨 Starting generation...")

    width, height = parse_dimensions(dimensions_str)
    logs.append(f"Requested dimensions: {width}x{height}")

    # Dimension compatibility warnings
    if loaded_model_type == "sd1.5":
        if width > 768 or height > 768:
            logs.append(f"Warning: SD 1.5 with large dimensions ({width}x{height}) may be unstable. Max 768px recommended.")
        elif width < 512 or height < 512 : # SD1.5 typically trained at 512
             logs.append(f"Warning: SD 1.5 with small dimensions ({width}x{height}). Results may vary from optimal 512px.")
    elif loaded_model_type == "sdxl":
        if width < 768 or height < 768 :
             logs.append(f"Warning: SDXL at small dimensions ({width}x{height}). Suboptimal results likely below 768px (1024px ideal).")
        if not torch.cuda.is_available() and (width >=1024 or height >=1024):
             logs.append(f"CRITICAL WARNING: SDXL at {width}x{height} on CPU will be extremely slow and may fail due to memory.")


    styled_prompt, style_negative_prefix = apply_style(prompt, style_name)
    final_negative_prompt = negative_prompt
    if style_negative_prefix:
        final_negative_prompt = f"{style_negative_prefix}, {negative_prompt}" if negative_prompt else style_negative_prefix

    progress(0.05, desc=f"📝 Prompting model...")
    print(f"Generating: '{styled_prompt}' ({width}x{height}), Neg: '{final_negative_prompt}'")

    try:
        seed = int(seed_value)
        if seed == -1: seed = get_random_seed()
    except (ValueError, TypeError):
        seed = get_random_seed()
    logs.append(f"Seed: {seed}")
    print(f"Seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)

    generation_args = {
        "prompt": styled_prompt, "negative_prompt": final_negative_prompt or None,
        "num_inference_steps": int(num_inference_steps), "guidance_scale": float(guidance_scale),
        "generator": generator, "width": width, "height": height
    }

    generated_image = None
    try:
        # Use autocast for mixed precision if on CUDA and float16 is active in pipe
        # pipe.torch_dtype check is important if float16 was disabled during load for some reason
        use_autocast = torch.cuda.is_available() and hasattr(pipe, "torch_dtype") and pipe.torch_dtype == torch.float16

        with torch.autocast("cuda", enabled=use_autocast):
            # Simulate progress for the diffusion steps more accurately
            # Diffusers pipelines don't have a direct step-by-step progress callback for Gradio easily.
            # This loop is a conceptual representation for the progress bar.
            # The actual call to `pipe()` happens once.
            for i in range(int(num_inference_steps)):
                # Update progress: diffusion part takes up to 80% of the bar
                current_progress_val = (i + 1) / int(num_inference_steps) * 0.8
                progress(current_progress_val, desc="🌀 Diffusing...")
            # Actual generation call after loop (progress bar is just a simulation of steps)
            generated_image = pipe(**generation_args).images[0]


        if generated_image is None: raise RuntimeError("Diffusion pipeline failed to return an image.")
        logs.append(f"Initial image generated: {generated_image.width}x{generated_image.height}")
        progress(0.8, desc="🎉 Initial image ready!")

        final_image = generated_image
        if upscale_active:
            if not REALESRGAN_AVAILABLE:
                logs.append("❌ Upscaling skipped: RealESRGAN library not installed.")
            elif upscaler_instance is None:
                logs.append("Upscaler model not loaded. Attempting to load for upscaling...")
                print("Upscaler model not loaded. Attempting to load for upscaling...")
                if not load_upscaler(print):
                    logs.append("❌ Upscaling skipped: Upscaler model failed to load.")
                else:
                    logs.append("Upscaler model loaded for this request.")

            if REALESRGAN_AVAILABLE and upscaler_instance is not None:
                if not torch.cuda.is_available():
                    logs.append("⏳ WARNING: Upscaling on CPU will be very slow!")
                logs.append("🚀 Upscaling image...")
                print("Upscaling image...")
                # Pass a lambda for progress to map it to the remaining 0.8-1.0 range
                final_image = upscale_image(generated_image, lambda p, desc_str: progress(0.8 + p * 0.2, desc=desc_str))
                logs.append(f"Image upscaled to: {final_image.width}x{final_image.height}")
            # If still no upscaler_instance after trying to load
            elif upscale_active and upscaler_instance is None and REALESRGAN_AVAILABLE :
                 logs.append("❌ Upscaling skipped: Upscaler model was not available/loaded for this session.")


        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sane_prompt_words = "".join(c if c.isalnum() or c.isspace() else " " for c in prompt).split()
        prompt_filename_prefix = "_".join(sane_prompt_words[:5])[:30] or "image"

        filename_prefix = custom_filename_prefix.strip() if custom_filename_prefix.strip() else prompt_filename_prefix
        final_filename = f"{filename_prefix}_{final_image.width}x{final_image.height}_{seed}_{timestamp}.png"
        logs.append(f"Suggested Filename: {final_filename}")

        info_text = (f"🖼️ Final Size: {final_image.width}x{final_image.height}\n"
                     f"🌱 Seed: {seed}\n"
                     f"🕒 Timestamp: {timestamp}\n"
                     f"🔧 Model: {current_model_id} ({loaded_model_type})\n"
                     f"🎨 Style: {style_name}\n"
                     f"📛 Filename: {final_filename}\n\n"
                     f"**Logs:**\n" + "\n".join(logs))

        progress(1.0, desc="✅ All Done!")
        print(f"Generation complete. Filename: {final_filename}")
        return final_image, info_text, final_filename

    except Exception as e:
        print(f"Error during image generation/upscaling: {e}")
        traceback.print_exc()
        logs.append(f"❌ ERROR: {e}")
        # Try to return partial logs if generation failed mid-way
        error_info_text = ("**Logs:**\n" + "\n".join(logs)) if logs else f"❌ ERROR: {e}"
        return None, error_info_text, None


# --- Gradio Interface ---
def create_gradio_interface(args):
    global current_model_id, loaded_model_type

    # Determine default dimension based on (potentially already) loaded model type
    # This happens after initial load attempt in __main__
    default_dimension = "512x512" # Fallback default
    if loaded_model_type == "sdxl":
        default_dimension = "1024x1024"
    elif loaded_model_type == "sd1.5":
        default_dimension = "512x512"

    dimension_choices = ["512x512", "512x768", "768x512", "768x768", "1024x768", "768x1024", "1024x1024"]


    theme = gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)
    css = """
        .gradio-container { max-width: 98% !important; padding: 15px; }
        footer { display: none !important }
        .gr-panel {
            border-radius: 10px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            padding:15px !important;
            background-color: #f9fafb;
        }
        #generate_button_main { min-height: 50px; font-size: 18px !important; }
        .status-warning { color: orange; font-weight: bold; }
        .status-critical { color: red; font-weight: bold; }
    """

    with gr.Blocks(theme=theme, css=css) as demo:
        # Determine initial title and status message based on actual loaded model
        # This is after the initial load attempt in __main__
        effective_model_id_for_title = current_model_id or "No Model Loaded"
        if not torch.cuda.is_available():
            effective_model_id_for_title += " (CPU Mode - Slow!)"

        gr.Markdown(f"# ✨ Stable Diffusion UI ✨ ({effective_model_id_for_title})")

        status_lines = [] # Will be populated by initial load messages
        # Placeholder, will be updated after initial load in __main__
        status_textbox = gr.Textbox(label="Status", value="Initializing...", interactive=False, lines=3)

        # CPU/GPU Warnings
        if not torch.cuda.is_available():
            gr.Markdown("<h3 class='status-critical'>⚠️ Running in CPU Mode! ⚠️</h3>"
                        "<p>No GPU detected. Image generation and upscaling will be **extremely slow**. "
                        "For reasonable performance, use a GPU-enabled environment (e.g., upgraded Hugging Face Space, local GPU, or Colab with GPU runtime).</p>"
                        "<p>SDXL models are effectively unusable on CPU. Defaulting to SD1.5 if SDXL was preferred.</p>")
        elif loaded_model_type == "sdxl" and "T4" in torch.cuda.get_device_name(0) if torch.cuda.is_available() else "": # Example check for T4
             gr.Markdown("<h3 class='status-warning'>💡 SDXL on T4 GPU Info 💡</h3>"
                        "<p>You are running an SDXL model on a T4 GPU. While possible, generations at 1024x1024 might be slow or hit VRAM limits, especially with upscaling. "
                        "Consider 768x768 or 512x_ for faster results if issues occur.</p>")


        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=480, elem_classes="gr-panel"):
                gr.Markdown("## ⚙️ Input Controls")
                with gr.Group():
                    prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your creative vision...")
                    negative_prompt_input = gr.Textbox(label="Negative Prompt", lines=2, placeholder="Elements to avoid...")

                with gr.Row():
                    with gr.Column(scale=1):
                        style_dropdown = gr.Dropdown(label="Artistic Style", choices=["None"] + list(STYLE_PRESETS.keys()), value="None")
                    with gr.Column(scale=1):
                        dimensions_dropdown = gr.Dropdown(label="Dimensions (W x H)", choices=dimension_choices, value=default_dimension,
                                                        info="SD1.5 best <=768px. SDXL best >=768px (1024px ideal).")
                with gr.Row():
                    inference_steps_slider = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps") # Default 20 for faster CPU
                    cfg_scale_slider = gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.5, label="CFG Scale (Guidance)")

                with gr.Row():
                    seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    random_seed_button = gr.Button("🎲 Randomize", scale=1, min_width=50)

                with gr.Accordion("Output & Advanced Options", open=True):
                    custom_filename_input = gr.Textbox(label="Custom Filename Prefix", placeholder="my_creation_")
                    upscale_checkbox = gr.Checkbox(label="✨ Upscale Image (Real-ESRGAN x4)", value=args.default_upscale, visible=REALESRGAN_AVAILABLE,
                                                 info="Increases generation time. Very slow on CPU.")
                    if not REALESRGAN_AVAILABLE:
                        gr.Markdown("<p class='status-warning'>Upscaling disabled: RealESRGAN library not found.</p>")

                    if args.allow_model_change_ui:
                         new_model_id_input = gr.Textbox(label="Load New Diffusion Model ID", value=current_model_id or args.model_id,
                                                         placeholder="e.g., stabilityai/sdxl-base-1.0")
                         change_model_button = gr.Button("🔄 Load Model")

                generate_button = gr.Button("🖼️ Generate Image", variant="primary", elem_id="generate_button_main")

            with gr.Column(scale=3, min_width=520, elem_classes="gr-panel"):
                gr.Markdown("## 🖼️ Generated Image")
                with gr.Group() as output_display_group:
                    image_output = gr.Image(label="Output Image", type="pil", height=512, show_label=False, show_download_button=False, visible=False)
                    download_button = gr.DownloadButton(label="📥 Download Image", visible=False)

                info_output = gr.Textbox(label="Generation Info & Logs", lines=8, interactive=False, max_lines=15)

        # --- Event Handlers ---
        initial_status_messages_for_ui = []
        def update_status_for_gradio(message):
            initial_status_messages_for_ui.append(message)
            # This function is called during initial load.
            # The status_textbox will be updated once before demo.launch()
            print(f"[Initial Load Status] {message}")

        # This is where initial model loading happens for app.py
        # The results will be used to set the initial state of status_textbox.
        if not load_model(args.model_id, args.use_float16, args.attention_slicing, update_status_for_gradio):
            update_status_for_gradio(f"CRITICAL: Failed to load main diffusion model '{current_model_id or args.model_id}'. App may not function.")
        else:
            update_status_for_gradio(f"Main diffusion model '{current_model_id}' loaded ({loaded_model_type}).")
            # Update default dimension based on loaded model type for UI
            if loaded_model_type == "sdxl" and dimensions_dropdown.value != "1024x1024":
                dimensions_dropdown.value = "1024x1024" # Update component's initial value
                update_status_for_gradio("Adjusted default dimensions to 1024x1024 for SDXL.")
            elif loaded_model_type == "sd1.5" and dimensions_dropdown.value != "512x512":
                dimensions_dropdown.value = "512x512"
                update_status_for_gradio("Adjusted default dimensions to 512x512 for SD1.5.")

        if REALESRGAN_AVAILABLE:
            if not load_upscaler(update_status_for_gradio):
                update_status_for_gradio("Warning: Failed to load Real-ESRGAN upscaler. Upscaling will be unavailable.")
            else:
                update_status_for_gradio("Real-ESRGAN upscaler loaded.")
        else:
            update_status_for_gradio("RealESRGAN library not found. Upscaling is disabled.")

        status_textbox.value = "\n".join(initial_status_messages_for_ui) # Set final initial status on the textbox


        def on_generate_click_wrapper_app(prompt, neg_prompt, style, dimensions, steps, cfg, seed, filename_prefix, upscale_active_ui, progress=gr.Progress(track_ τότε=True)):
            # Warn about upscaling on CPU if selected
            current_status_val = status_textbox.value
            if upscale_active_ui and not torch.cuda.is_available() and REALESRGAN_AVAILABLE:
                warning_msg = "PERFORMANCE WARNING: Upscaling on CPU will be extremely slow!"
                # Prepend warning to status, or use a dedicated warning component
                yield { status_textbox: gr.update(value=f"{warning_msg}\n{current_status_val}\n⏳ Processing request...") }
            else:
                 yield { status_textbox: gr.update(value=f"{current_status_val}\n⏳ Processing request...") }


            yield {
                generate_button: gr.update(interactive=False, value="⏳ Working..."),
                image_output: gr.update(visible=False, value=None),
                download_button: gr.update(visible=False),
                info_output: "Starting generation process...",
            }

            img, info_text_result, generated_filename = generate_image_fn(
                prompt, neg_prompt, style, dimensions, steps, cfg, seed,
                filename_prefix, upscale_active_ui, progress=progress
            )

            dl_button_update = gr.DownloadButton.update(visible=False)
            if img and generated_filename:
                temp_dir = "temp_app_generated_images" # Changed dir name
                os.makedirs(temp_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_dir, generated_filename)
                try:
                    img.save(temp_file_path)
                    dl_button_update = gr.DownloadButton.update(value=temp_file_path, label=f"📥 Download ({generated_filename})", visible=True)
                except Exception as e_save:
                    info_text_result += f"\nError saving temp file for download: {e_save}"

            yield {
                image_output: gr.update(value=img, visible=True if img else False),
                info_output: info_text_result,
                generate_button: gr.update(interactive=True, value="🖼️ Generate Image"),
                status_textbox: gr.update(value=f"{status_textbox.value.split('⏳ Processing request...')[0]}\n✅ Process complete." if img else f"{status_textbox.value.split('⏳ Processing request...')[0]}\n❌ Process failed. Check logs."),
                download_button: dl_button_update,
            }

        generate_button.click(
            fn=on_generate_click_wrapper_app,
            inputs=[prompt_input, negative_prompt_input, style_dropdown, dimensions_dropdown,
                    inference_steps_slider, cfg_scale_slider, seed_input,
                    custom_filename_input, upscale_checkbox],
            outputs=[status_textbox, generate_button, image_output, download_button, info_output],
        )

        random_seed_button.click(fn=lambda: gr.update(value=get_random_seed()), inputs=None, outputs=seed_input)

        if args.allow_model_change_ui and 'change_model_button' in locals():
            def handle_change_model_click(new_model_id_str_ui):
                global pipe, current_model_id, loaded_model_type

                initial_status_msg = status_textbox.value.split("\n")[0] # Preserve first line of status
                yield {status_textbox: gr.update(value=f"{initial_status_msg}\nAttempting to change model to {new_model_id_str_ui}..."),
                       change_model_button: gr.update(interactive=False),
                       generate_button: gr.update(interactive=False)}

                pipe = None
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                # Use a lambda to append to the status_textbox for live updates during model load
                # This is more complex than direct print, but shows progress in UI
                temp_load_msgs = [f"{initial_status_msg}\nAttempting to change model to {new_model_id_str_ui}..."]
                def ui_status_updater(msg):
                    temp_load_msgs.append(msg)
                    # This yield inside a non-generator won't work as expected for live UI updates
                    # Instead, we collect messages and update at the end of this event.
                    print(f"[Model Change] {msg}")


                success = load_model(new_model_id_str_ui, args.use_float16, args.attention_slicing, ui_status_updater)

                final_status_msg = f"Model '{current_model_id}' loaded ({loaded_model_type})." if success else f"Failed to load model '{new_model_id_str_ui}'."
                temp_load_msgs.append(final_status_msg)

                yield {status_textbox: gr.update(value="\n".join(temp_load_msgs)),
                       change_model_button: gr.update(interactive=True),
                       generate_button: gr.update(interactive=True),
                       new_model_id_input: gr.update(value=current_model_id if success else new_model_id_str_ui)
                      }

            change_model_button.click(
                handle_change_model_click,
                inputs=[new_model_id_input],
                outputs=[status_textbox, change_model_button, generate_button, new_model_id_input]
            )

    return demo

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gradio app for Ultra-Professional Stable Diffusion.")
    # Default model is now SD1.5 for CPU HF Spaces compatibility
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_FOR_CPU_TIER,
                        help=f"Preferred HuggingFace model ID (default for CPU: {DEFAULT_MODEL_FOR_CPU_TIER}, for GPU consider {SDXL_MODEL_ID_OPTION})")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link for public access")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (overrides CUDA detection)")

    # BooleanOptionalAction requires Python 3.9+
    # For broader compatibility, use store_true/store_false or type=lambda x: x.lower() == 'true'
    parser.add_argument("--use_float16", type=str, default="true", choices=['true', 'false'],
                        help="Enable float16 precision (default: true). Set to 'false' to disable.")
    parser.add_argument("--attention_slicing", type=str, default="true", choices=['true', 'false'],
                        help="Enable attention slicing (default: true). Set to 'false' to disable.")

    parser.add_argument("--allow_model_change_ui", action="store_true", help="Allow changing diffusion model via UI (experimental)")
    # Default upscale to False for CPU tier
    parser.add_argument("--default_upscale", type=str, default="false", choices=['true', 'false'],
                        help="Default state for upscaling checkbox (default: false for CPU tier). Set to 'true' to enable.")

    args = parser.parse_args()

    # Convert string bool args to Python bools
    args.use_float16 = args.use_float16.lower() == 'true'
    args.attention_slicing = args.attention_slicing.lower() == 'true'
    args.default_upscale = args.default_upscale.lower() == 'true'


    if args.cpu or not torch.cuda.is_available():
        args.use_float16 = False # Float16 is typically for CUDA
        print("Running in CPU Mode. Float16 disabled. Performance will be very slow.")
        if "sdxl" in args.model_id.lower():
            print(f"WARNING: Requested model '{args.model_id}' is SDXL but running on CPU. This is highly discouraged and may fail or be unusable. Consider using an SD1.5 model like '{DEFAULT_MODEL_FOR_CPU_TIER}'.")

    print("--- Application Configuration ---")
    print(f"Requested Model ID: {args.model_id}")
    # The actual loaded model (current_model_id) will be determined by load_model logic
    print(f"Device: {'CPU' if args.cpu or not torch.cuda.is_available() else 'CUDA'}")
    if torch.cuda.is_available() and not args.cpu:
        print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Use float16: {args.use_float16 if torch.cuda.is_available() and not args.cpu else 'N/A (CPU or disabled)'}")
    print(f"Attention Slicing: {args.attention_slicing if torch.cuda.is_available() and not args.cpu else 'N/A (CPU or disabled)'}")
    print(f"Allow Model Change in UI: {args.allow_model_change_ui}")
    print(f"Default Upscale State: {args.default_upscale}")
    print(f"RealESRGAN Library Available: {REALESRGAN_AVAILABLE}")
    print("-------------------------------")

    # Initial model loading and UI creation is now handled within create_gradio_interface
    # to allow the UI to reflect the status of these operations better.
    gradio_ui = create_gradio_interface(args)

    print(f"Launching Gradio app on port {args.port}. Public link via --share: {args.share}")
    # For Hugging Face Spaces, share=True is often handled by the Space runtime if not set.
    # Binding to 0.0.0.0 is important for Docker/Spaces.
    gradio_ui.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, debug=False)

print("app.py script finished.")
