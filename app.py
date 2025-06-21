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
    print("To enable upscaling, please install realesrgan and basicsr: pip install realesrgan basicsr")
    REALESRGAN_AVAILABLE = False
    RealESRGANer = None # Define for type hinting if needed, but it won't be used
    RRDBNet = None


# --- Configuration & Global State ---
# Main Diffusion Model
pipe = None
current_model_id = None
loaded_model_type = None # 'sdxl', 'sd1.5', or None

PREFERRED_SDXL_MODEL_ID = "stabilityai/sdxl-base-1.0"
FALLBACK_SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
# User can override preferred model via CLI --model_id, which will be attempted first.
# If that fails and it was an SDXL model, it will then try FALLBACK_SD15_MODEL_ID.

# Upscaler Model
upscaler_instance = None # Renamed from upscaler_model to avoid conflict if imported elsewhere
REALESRGAN_MODEL_NAME = 'RealESRGAN_x4plus'
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
def load_diffusion_pipeline(model_id_to_load, is_sdxl, use_float16, use_attention_slicing, status_update_fn):
    global pipe # Modifies the global pipe instance

    pipeline_args = {}
    if torch.cuda.is_available() and use_float16:
        pipeline_args["torch_dtype"] = torch.float16
        status_update_fn(f"Using torch_dtype: float16 for {model_id_to_load}")

    if is_sdxl:
        status_update_fn(f"Attempting to load SDXL model: {model_id_to_load}...")
        pipe = AutoPipelineForText2Image.from_pretrained(model_id_to_load, **pipeline_args)
    else: # SD 1.5 or other non-SDXL
        status_update_fn(f"Attempting to load non-SDXL model: {model_id_to_load}...")
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id_to_load, subfolder="scheduler")
        pipeline_args["scheduler"] = scheduler
        pipe = StableDiffusionPipeline.from_pretrained(model_id_to_load, **pipeline_args)

    if torch.cuda.is_available():
        status_update_fn(f"Moving {model_id_to_load} to CUDA...")
        pipe.to("cuda")

    if use_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        status_update_fn(f"Enabling attention slicing for {model_id_to_load}.")
        pipe.enable_attention_slicing()

    status_update_fn(f"Model {model_id_to_load} loaded successfully.")
    return model_id_to_load, "sdxl" if is_sdxl else "sd1.5"


def load_model(preferred_model_id, use_float16=True, use_attention_slicing=False, status_update_fn=None):
    global pipe, current_model_id, loaded_model_type

    if status_update_fn is None: status_update_fn = print # Default to print if no UI update fn

    if pipe is not None and current_model_id == preferred_model_id:
        status_update_fn(f"Model '{preferred_model_id}' already loaded.")
        return True

    # Determine if preferred model is SDXL
    is_preferred_sdxl = "sdxl" in preferred_model_id.lower()

    try:
        # Attempt to load the preferred model
        loaded_id, type_loaded = load_diffusion_pipeline(preferred_model_id, is_preferred_sdxl, use_float16, use_attention_slicing, status_update_fn)
        current_model_id = loaded_id
        loaded_model_type = type_loaded
        return True
    except Exception as e_preferred:
        status_update_fn(f"Failed to load preferred model '{preferred_model_id}': {e_preferred}")
        traceback.print_exc()
        pipe = None # Ensure pipe is reset

        if is_preferred_sdxl: # If preferred was SDXL and failed, try fallback SD1.5
            status_update_fn(f"Attempting to load fallback SD 1.5 model: {FALLBACK_SD15_MODEL_ID}...")
            try:
                loaded_id, type_loaded = load_diffusion_pipeline(FALLBACK_SD15_MODEL_ID, False, use_float16, use_attention_slicing, status_update_fn)
                current_model_id = loaded_id
                loaded_model_type = type_loaded
                return True
            except Exception as e_fallback:
                status_update_fn(f"Failed to load fallback SD 1.5 model '{FALLBACK_SD15_MODEL_ID}': {e_fallback}")
                traceback.print_exc()
                current_model_id = None
                loaded_model_type = None
                return False
        else: # Preferred was not SDXL and it failed
            current_model_id = None
            loaded_model_type = None
            return False

# --- Upscaler Functions ---
def load_upscaler(status_update_fn=None):
    global upscaler_instance
    if not REALESRGAN_AVAILABLE:
        if status_update_fn: status_update_fn("RealESRGAN library not available. Upscaling disabled.")
        return False

    if upscaler_instance is not None:
        if status_update_fn: status_update_fn("Upscaler model already loaded.")
        return True

    if status_update_fn: status_update_fn(f"Loading Real-ESRGAN upscaler: {REALESRGAN_MODEL_NAME}...")
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=REALESRGAN_SCALE)
        half_precision = torch.cuda.is_available()

        upscaler_instance = RealESRGANer(
            scale=REALESRGAN_SCALE, model_path=None, model=model, dni_weight=None,
            model_name=REALESRGAN_MODEL_NAME, tile=0, tile_pad=10, pre_pad=0,
            half=half_precision, gpu_id=0 if torch.cuda.is_available() else None
        )
        if status_update_fn: status_update_fn(f"Real-ESRGAN '{REALESRGAN_MODEL_NAME}' loaded.")
        return True
    except Exception as e:
        if status_update_fn: status_update_fn(f"Error loading Real-ESRGAN: {e}. Upscaling disabled.")
        traceback.print_exc()
        upscaler_instance = None
        return False

def upscale_image(pil_image, progress_callback=None):
    global upscaler_instance
    if not REALESRGAN_AVAILABLE or upscaler_instance is None:
        print("Upscaling skipped: RealESRGAN not available or model not loaded.")
        return pil_image

    if progress_callback: progress_callback(0.0, desc="✨ Upscaling...")
    print(f"Upscaling image (original size: {pil_image.width}x{pil_image.height}) with {REALESRGAN_MODEL_NAME}...")

    # Convert PIL to OpenCV format (NumPy array BGR)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    try:
        output_cv, _ = upscaler_instance.enhance(cv_image, outscale=REALESRGAN_SCALE) # Pass scale explicitly
        # Convert OpenCV back to PIL format (RGB)
        output_pil = Image.fromarray(cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB))
        print(f"Image upscaled to: {output_pil.width}x{output_pil.height}")
        if progress_callback: progress_callback(1.0, desc="✨ Upscaling Complete!")
        return output_pil
    except Exception as e:
        print(f"Error during upscaling: {e}")
        traceback.print_exc()
        if progress_callback: progress_callback(1.0, desc="❌ Upscaling Failed.")
        return pil_image # Return original on error


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

    if loaded_model_type == "sd1.5" and (width > 768 or height > 768):
        logs.append(f"Warning: SD 1.5 works best <= 768px. Large dimensions ({width}x{height}) might be unstable. Consider 512x512 or 512x768.")
        # Optionally, auto-adjust: width, height = min(width, 768), min(height, 768)
    elif loaded_model_type == "sdxl" and (width < 768 or height < 768):
        logs.append(f"Warning: SDXL performs best >= 768px (ideally 1024px). Small dimensions ({width}x{height}) might yield suboptimal results.")

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
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            for i in range(int(num_inference_steps)):
                progress((i + 1) / int(num_inference_steps) * 0.8, desc="🌀 Diffusing...") # Diffusion takes 80% of progress
                if i == int(num_inference_steps) - 1: # Generate on last conceptual step for progress bar
                     generated_image = pipe(**generation_args).images[0]
            if generated_image is None and int(num_inference_steps) == 0: # Edge case
                 generated_image = pipe(**generation_args).images[0]

        if generated_image is None: raise RuntimeError("Diffusion pipeline failed to return an image.")
        logs.append(f"Initial image generated: {generated_image.width}x{generated_image.height}")
        progress(0.8, desc="🎉 Initial image ready!")

        final_image = generated_image
        if upscale_active:
            if not REALESRGAN_AVAILABLE:
                logs.append("❌ Upscaling skipped: RealESRGAN library not installed.")
            elif upscaler_instance is None: # Attempt to load upscaler if not already loaded
                logs.append("Upscaler model not loaded. Attempting to load for upscaling...")
                print("Upscaler model not loaded. Attempting to load for upscaling...")
                if not load_upscaler(print): # Pass print as simple status_update_fn
                    logs.append("❌ Upscaling skipped: Upscaler model failed to load.")
                else:
                    logs.append("Upscaler model loaded for this request.")

            if REALESRGAN_AVAILABLE and upscaler_instance is not None:
                logs.append("🚀 Upscaling image...")
                print("Upscaling image...")
                # Pass a lambda for progress to map it to the remaining 0.8-1.0 range
                final_image = upscale_image(generated_image, lambda p, desc: progress(0.8 + p * 0.2, desc=desc))
                logs.append(f"Image upscaled to: {final_image.width}x{final_image.height}")
            else: # If still no upscaler_instance
                 logs.append("❌ Upscaling skipped: Upscaler model not available/loaded.")


        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sane_prompt_words = "".join(c if c.isalnum() or c.isspace() else " " for c in prompt).split()
        prompt_filename_prefix = "_".join(sane_prompt_words[:5])[:30] or "image"

        filename_prefix = custom_filename_prefix.strip() if custom_filename_prefix.strip() else prompt_filename_prefix
        final_filename = f"{filename_prefix}_{final_image.width}x{final_image.height}_{seed}_{timestamp}.png"
        logs.append(f"Suggested Filename: {final_filename}")

        # For gr.Image's download button, it saves the PIL image automatically.
        # If using a separate gr.DownloadButton, we'd save `final_image` to a temp path and return the path.

        info_text = (f"🖼️ Final Size: {final_image.width}x{final_image.height}\n"
                     f"🌱 Seed: {seed}\n"
                     f"🕒 Timestamp: {timestamp}\n"
                     f"🔧 Model: {current_model_id} ({loaded_model_type})\n"
                     f"🎨 Style: {style_name}\n"
                     f"📛 Filename: {final_filename}\n\n"
                     f"**Logs:**\n" + "\n".join(logs))

        progress(1.0, desc="✅ All Done!")
        print(f"Generation complete. Filename: {final_filename}")
        return final_image, info_text, final_filename # Return filename for potential use with DownloadButton

    except Exception as e:
        print(f"Error during image generation/upscaling: {e}")
        traceback.print_exc()
        logs.append(f"❌ ERROR: {e}")
        return None, "\n".join(logs), None


# --- Gradio Interface ---
def create_gradio_interface(args):
    global current_model_id, loaded_model_type # To display in UI title/status

    dimension_choices = ["512x512", "512x768", "768x512", "768x768", "1024x1024", "768x1024", "1024x768"]
    default_dimension = "512x512"
    if loaded_model_type == "sdxl": default_dimension = "1024x1024"


    theme = gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)
    css = """
        .gradio-container { max-width: 98% !important; padding: 15px; }
        footer { display: none !important }
        .gr-panel {
            border-radius: 10px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            padding:15px !important;
            background-color: #f9fafb; /* Light grey background for panels */
        }
        #generate_button_main { min-height: 50px; font-size: 18px !important; }
    """

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(f"# ✨ Ultra-Professional Stable Diffusion UI ✨ ({args.model_id})")
        status_textbox = gr.Textbox(label="Status", value="Loading models...", interactive=False, lines=2)

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
                                                        info="SD1.5 best <=768px. SDXL best >=768px (ideally 1024px).")
                with gr.Row():
                    inference_steps_slider = gr.Slider(minimum=10, maximum=50, value=25, step=1, label="Inference Steps")
                    cfg_scale_slider = gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.5, label="CFG Scale (Guidance)")

                with gr.Row():
                    seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    random_seed_button = gr.Button("🎲 Randomize", scale=1, min_width=50)

                with gr.Accordion("Output & Advanced Options", open=True):
                    custom_filename_input = gr.Textbox(label="Custom Filename Prefix", placeholder="my_masterpiece_")
                    upscale_checkbox = gr.Checkbox(label="✨ Upscale Image (Real-ESRGAN x4)", value=args.default_upscale, visible=REALESRGAN_AVAILABLE,
                                                 info="Increases generation time. Requires RealESRGAN.")
                    if not REALESRGAN_AVAILABLE:
                        gr.Markdown("<p style='color:orange;'>Upscaling disabled: RealESRGAN library not found.</p>")

                    if args.allow_model_change_ui: # Experimental model change
                         new_model_id_input = gr.Textbox(label="Load New Diffusion Model ID", value=current_model_id or args.model_id)
                         change_model_button = gr.Button("🔄 Load Model")


                generate_button = gr.Button("🖼️ Generate Image", variant="primary", elem_id="generate_button_main")

            with gr.Column(scale=3, min_width=520, elem_classes="gr-panel"):
                gr.Markdown("## 🖼️ Generated Image")
                with gr.Group() as output_display_group: # For spinner target
                    image_output = gr.Image(label="Output Image", type="pil", height=512, show_label=False, show_download_button=False, visible=False)
                    # We use a separate download button to control filename
                    download_button = gr.DownloadButton(label="📥 Download Image", visible=False)

                info_output = gr.Textbox(label="Generation Info & Logs", lines=8, interactive=False, max_lines=15)

        # --- Event Handlers ---
        def update_status_fn(message): # Used by model loaders
            status_textbox.value = message # This might not update live if called outside Gradio event. Print is primary.
            print(f"[Status Update] {message}")
            return gr.update(value=message) # Return update for Gradio event chains

        # Initial model loads (diffusion and upscaler)
        # These are called when the script starts, before Gradio launch.
        # The status_textbox will show the *final* status of these loads.
        initial_load_messages = []
        if not load_model(args.model_id, args.use_float16, args.attention_slicing, lambda msg: initial_load_messages.append(msg)):
            initial_load_messages.append(f"CRITICAL: Failed to load main diffusion model '{current_model_id or args.model_id}'. App may not function.")
        else:
            initial_load_messages.append(f"Main diffusion model '{current_model_id}' loaded ({loaded_model_type}).")
            # Update default dimension based on loaded model type for UI
            if loaded_model_type == "sdxl" and default_dimension != "1024x1024":
                dimensions_dropdown.value = "1024x1024" # This direct update might not reflect. Better to set in definition.
                initial_load_messages.append("Adjusted default dimensions to 1024x1024 for SDXL.")
            elif loaded_model_type == "sd1.5" and default_dimension != "512x512":
                dimensions_dropdown.value = "512x512"
                initial_load_messages.append("Adjusted default dimensions to 512x512 for SD1.5.")


        if REALESRGAN_AVAILABLE: # Only try to load if library is present
            if not load_upscaler(lambda msg: initial_load_messages.append(msg)):
                initial_load_messages.append("Warning: Failed to load Real-ESRGAN upscaler model. Upscaling will be unavailable.")
            else:
                initial_load_messages.append("Real-ESRGAN upscaler loaded.")
        else:
            initial_load_messages.append("RealESRGAN library not found. Upscaling is disabled.")

        status_textbox.value = "\n".join(initial_load_messages) # Set final initial status

        # Update UI title with loaded model
        demo.title = f"✨ Ultra-Professional Stable Diffusion UI ✨ ({current_model_id or 'No Model Loaded'})" # This might not work post-init
        # A gr.Markdown element for the title would be more reliably updatable.

        def on_generate_click_wrapper_app(prompt, neg_prompt, style, dimensions, steps, cfg, seed, filename_prefix, upscale_active_ui, progress=gr.Progress(track_ τότε=True)):
            yield { # Initial UI updates for loading
                generate_button: gr.update(interactive=False, value="⏳ Working..."),
                status_textbox: gr.update(value="⏳ Processing request..."),
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
                # Save image to a temporary path for the download button
                temp_dir = "temp_app_images"
                os.makedirs(temp_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_dir, generated_filename)
                try:
                    img.save(temp_file_path)
                    dl_button_update = gr.DownloadButton.update(value=temp_file_path, label=f"📥 Download ({generated_filename})", visible=True)
                except Exception as e_save:
                    info_text_result += f"\nError saving temp file for download: {e_save}"

            yield { # Final UI updates
                image_output: gr.update(value=img, visible=True if img else False),
                info_output: info_text_result,
                generate_button: gr.update(interactive=True, value="🖼️ Generate Image"),
                status_textbox: gr.update(value="✅ Process complete." if img else "❌ Process failed. Check logs."),
                download_button: dl_button_update,
            }

        generate_button.click(
            fn=on_generate_click_wrapper_app,
            inputs=[prompt_input, negative_prompt_input, style_dropdown, dimensions_dropdown,
                    inference_steps_slider, cfg_scale_slider, seed_input,
                    custom_filename_input, upscale_checkbox],
            outputs=[generate_button, status_textbox, image_output, download_button, info_output],
            # target output_display_group for loading status implicitly by its children being outputs
        )

        random_seed_button.click(fn=lambda: gr.update(value=get_random_seed()), inputs=None, outputs=seed_input)

        if args.allow_model_change_ui and 'change_model_button' in locals():
            def handle_change_model_click(new_model_id_str_ui):
                global pipe, current_model_id, loaded_model_type # Ensure globals are modified
                yield {status_textbox: gr.update(value=f"Attempting to change model to {new_model_id_str_ui}..."),
                       change_model_button: gr.update(interactive=False),
                       generate_button: gr.update(interactive=False)}

                pipe = None # Release current model
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                success = load_model(new_model_id_str_ui, args.use_float16, args.attention_slicing, update_status_fn)

                final_status = f"Model '{current_model_id}' loaded ({loaded_model_type})." if success else f"Failed to load model '{new_model_id_str_ui}'."
                yield {status_textbox: gr.update(value=final_status),
                       change_model_button: gr.update(interactive=True),
                       generate_button: gr.update(interactive=True),
                       # Update the new_model_id_input to reflect actual loaded model or failed attempt
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
    parser.add_argument("--model_id", type=str, default=PREFERRED_SDXL_MODEL_ID, help=f"Preferred HuggingFace model ID (default: {PREFERRED_SDXL_MODEL_ID})")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (very slow)")
    parser.add_argument("--use_float16", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable float16 precision (default: enabled). Use --no-use_float16 to disable.")
    parser.add_argument("--attention_slicing", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable attention slicing (default: enabled).")
    parser.add_argument("--allow_model_change_ui", action="store_true", help="Allow changing diffusion model via UI (experimental)")
    parser.add_argument("--default_upscale", action=argparse.BooleanOptionalAction, default=True, help="Default state for upscaling checkbox (default: enabled).")


    args = parser.parse_args()

    if args.cpu: # Override GPU options if CPU is forced
        args.use_float16 = False
        print("CPU mode forced. Float16 disabled.")

    print("--- Configuration ---")
    print(f"Preferred Model ID: {args.model_id}")
    print(f"Fallback SD1.5 Model ID: {FALLBACK_SD15_MODEL_ID}")
    print(f"Device: {'CPU' if args.cpu or not torch.cuda.is_available() else 'CUDA'}")
    print(f"Use float16: {args.use_float16}")
    print(f"Attention Slicing: {args.attention_slicing}")
    print(f"Allow Model Change in UI: {args.allow_model_change_ui}")
    print(f"Default Upscale State: {args.default_upscale}")
    print(f"RealESRGAN Available: {REALESRGAN_AVAILABLE}")
    print("---------------------")

    # Gradio interface creation is now done after initial model load attempts are logged.
    # The create_gradio_interface function itself handles the initial load calls.
    gradio_ui = create_gradio_interface(args)

    print(f"Launching Gradio app on port {args.port}...")
    gradio_ui.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, debug=False)

print("app.py script finished.")
