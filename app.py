import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoPipelineForText2Image
from PIL import Image
import random
import argparse
import time
import os

# --- Configuration ---
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5" # Safer for free tier Colab
# DEFAULT_MODEL_ID = "stabilityai/sdxl-base-1.0" # Alternative, might be too heavy for free Colab
# For SDXL, you might need:
# from diffusers import AutoPipelineForText2Image

pipe = None
current_model_id = DEFAULT_MODEL_ID

# --- Style Presets ---
STYLE_PRESETS = {
    "None": {"prompt_suffix": "", "negative_prompt_prefix": ""},
    "Realistic": {"prompt_suffix": "photorealistic, 4k, ultra detailed, cinematic lighting", "negative_prompt_prefix": "cartoon, anime, drawing, sketch, stylized"},
    "Anime": {"prompt_suffix": "anime style, key visual, vibrant, beautiful, detailed, studio trigger, official art", "negative_prompt_prefix": "photorealistic, 3d render, ugly, disfigured"},
    "Fantasy Art": {"prompt_suffix": "fantasy art, detailed, intricate, epic, trending on artstation, by greg rutkowski, Brom", "negative_prompt_prefix": "photorealistic, modern, simple"},
    "Digital Painting": {"prompt_suffix": "digital painting, concept art, smooth, sharp focus, illustration", "negative_prompt_prefix": "photo, 3d model, realism"},
    "3D Render": {"prompt_suffix": "3d render, octane render, blender, detailed, physically based rendering", "negative_prompt_prefix": "2d, drawing, sketch, painting"},
}

# --- Helper Functions ---
def get_random_seed():
    return random.randint(0, 2**32 - 1)

def apply_style(prompt, style_name):
    if style_name == "None" or style_name not in STYLE_PRESETS:
        return prompt, "" # Return empty negative prefix if style is None

    preset = STYLE_PRESETS[style_name]
    return f"{prompt}, {preset['prompt_suffix']}", preset['negative_prompt_prefix']

# --- Model Loading ---
def load_model(model_id_to_load=DEFAULT_MODEL_ID, use_float16=True, use_attention_slicing=False, status_update_fn=None):
    global pipe, current_model_id
    if pipe is not None and current_model_id == model_id_to_load:
        if status_update_fn: status_update_fn(f"Model '{model_id_to_load}' already loaded.")
        print(f"Model '{model_id_to_load}' already loaded.")
        return

    if status_update_fn: status_update_fn(f"Loading model: {model_id_to_load}...")
    print(f"Loading model: {model_id_to_load}...")

    pipeline_args = {}
    if torch.cuda.is_available() and use_float16:
        pipeline_args["torch_dtype"] = torch.float16
        print("Using float16 precision.")

    # Determine which pipeline to use
    model_is_sdxl = "sdxl" in model_id_to_load.lower()

    try:
        if model_is_sdxl:
            # SDXL specific loading
            if status_update_fn: status_update_fn(f"Loading SDXL model: {model_id_to_load}...")
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_to_load, **pipeline_args)
            # SDXL Refiner (optional, adds complexity and VRAM)
            # refiner = AutoPipelineForImage2Image.from_pretrained(
            #     "stabilityai/sdxl-refiner-1.0",
            #     text_encoder_2=pipe.text_encoder_2,
            #     vae=pipe.vae,
            #     torch_dtype=torch.float16 if torch.cuda.is_available() and use_float16 else torch.float32,
            #     use_safetensors=True, variant="fp16" if use_float16 else None
            # )
            # if torch.cuda.is_available():
            #     refiner.to("cuda")
            # pipe.refiner = refiner # Attach refiner to the pipe
        else:
            # Standard Stable Diffusion pipeline
            if status_update_fn: status_update_fn(f"Loading Stable Diffusion model: {model_id_to_load}...")
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id_to_load, subfolder="scheduler")
            pipeline_args["scheduler"] = scheduler
            pipe = StableDiffusionPipeline.from_pretrained(model_id_to_load, **pipeline_args)

        if torch.cuda.is_available():
            print("Moving model to CUDA.")
            pipe = pipe.to("cuda")
        else:
            print("CUDA not available. Running on CPU (this will be very slow).")

        if use_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            print("Enabling attention slicing.")
            pipe.enable_attention_slicing()

        current_model_id = model_id_to_load
        if status_update_fn: status_update_fn(f"Model '{model_id_to_load}' loaded successfully.")
        print(f"Model '{model_id_to_load}' loaded successfully.")
    except Exception as e:
        if status_update_fn: status_update_fn(f"Error loading model '{model_id_to_load}': {e}")
        print(f"Error loading model '{model_id_to_load}': {e}")
        pipe = None
        current_model_id = None
        raise

# --- Image Generation ---
# Note: Added custom_filename_input to the signature
def generate_image_fn(prompt, negative_prompt, style_name, num_inference_steps, guidance_scale, seed_value, custom_filename_input="", progress=gr.Progress(track_ τότε=True)):
    global pipe
    if pipe is None:
        # Return structure matching expected outputs for on_generate_click_wrapper
        return None, "Model not loaded. Please wait or check logs.", gr.DownloadButton.update(visible=False)


    progress(0, desc="Starting image generation...")

    # Apply style preset
    styled_prompt, style_negative_prefix = apply_style(prompt, style_name)
    if style_negative_prefix and negative_prompt:
        final_negative_prompt = f"{style_negative_prefix}, {negative_prompt}"
    elif style_negative_prefix:
        final_negative_prompt = style_negative_prefix
    else:
        final_negative_prompt = negative_prompt

    progress(0.1, desc=f"Prompt: {styled_prompt[:100]}...") # Show beginning of prompt
    print(f"Generating image with prompt: '{styled_prompt}'")
    if final_negative_prompt:
        print(f"Negative prompt: '{final_negative_prompt}'")

    try:
        seed = int(seed_value)
    except (ValueError, TypeError):
        seed = get_random_seed()
    print(f"Using seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)

    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)

    generation_args = {
        "prompt": styled_prompt,
        "negative_prompt": final_negative_prompt if final_negative_prompt else None,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator
    }

    image = None # Initialize image variable
    try:
        if device == "cuda" and hasattr(pipe, 'torch_dtype') and pipe.torch_dtype == torch.float16:
            with torch.autocast("cuda"):
                for i in progress.tqdm(range(num_inference_steps), desc="Generating image"):
                    if i == num_inference_steps -1:
                        image = pipe(**generation_args).images[0]
        else:
            for i in progress.tqdm(range(num_inference_steps), desc="Generating image (CPU)"):
                if i == num_inference_steps -1:
                    image = pipe(**generation_args).images[0]

        if image is None: # Should be caught by an error earlier if pipe fails
            raise RuntimeError("Image generation did not produce an image.")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Sanitize prompt for filename (simple version)
        sane_prompt_prefix = "".join(c if c.isalnum() else "_" for c in styled_prompt[:30])

        # Use custom_filename_input if provided, otherwise default to sanitized prompt
        filename_prefix_to_use = custom_filename_input.strip() if custom_filename_input.strip() else sane_prompt_prefix

        base_filename = f"{filename_prefix_to_use}_{timestamp}_seed{seed}.png"

        print(f"Image generated. Suggested filename: {base_filename}")
        progress(1.0, desc="Image generation complete!")

        info_text = f"Seed: {seed}\nTimestamp: {timestamp}\nModel: {current_model_id}\nSuggested Filename: {base_filename}"

        # For app.py, gr.Image's own download button is used primarily.
        # The gr.DownloadButton.update here is more of a placeholder or for consistency if a separate button was the main download mechanism.
        # If gr.Image's show_download_button is true, it handles saving the PIL image to a temp file for its own button.
        # If we wanted a separate gr.DownloadButton to also work with this PIL image, we'd have to save 'image' to a path
        # and return that path for the 'value' of the DownloadButton.

        # For now, let gr.Image handle its download. The separate button is more for main.ipynb.
        # We return `image` (PIL object) for `gr.Image` and info.
        return image, info_text, gr.DownloadButton.update(visible=True) # This makes the button visible, but value is not set to a path

    except Exception as e:
        print(f"Error during image generation: {e}")
        import traceback
        traceback.print_exc()
        progress(1.0, desc=f"Error: {e}")
        return None, f"Error: {e}", gr.DownloadButton.update(visible=False)


# --- Gradio Interface Definition ---
def create_gradio_interface(initial_model_id, allow_model_change=True):

    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container { max-width: 95% !important; } footer {display: none !important}") as demo:
        gr.Markdown(f"# 🎨 Advanced Stable Diffusion UI (Model: {initial_model_id})")

        status_textbox = gr.Textbox(label="Status", value=f"Loading model: {initial_model_id}...", interactive=False, lines=1)

        with gr.Row(equal_height=False): # Allow columns to size independently
            # --- Left Column: Inputs ---
            with gr.Column(scale=2, min_width=400): # Input column
                gr.Markdown("## ⚙️ Input Controls")

                prompt_input = gr.Textbox(label="Enter your Prompt", lines=3, placeholder="e.g., A majestic lion in a futuristic city, neon lights, detailed fur")

                negative_prompt_input = gr.Textbox(label="Negative Prompt (what to avoid)", lines=2, placeholder="e.g., blurry, low quality, ugly, text, watermark, disfigured, cartoon")

                style_dropdown = gr.Dropdown(label="Artistic Style Preset", choices=["None"] + list(STYLE_PRESETS.keys())[1:], value="None")

                with gr.Row():
                    inference_steps_slider = gr.Slider(minimum=10, maximum=100, value=25, step=1, label="Inference Steps", info="More steps can improve detail but take longer.")
                    cfg_scale_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="CFG Scale (Guidance)", info="How strictly the prompt guides the image.")

                with gr.Row():
                    seed_input = gr.Number(label="Seed", value=get_random_seed(), precision=0, minimum=0)
                    random_seed_button = gr.Button("🎲 Randomize Seed", scale=1, min_width=50)

                generate_button = gr.Button("🖼️ Generate Image", variant="primary")

                with gr.Accordion("Output & Advanced Options", open=True): # Open by default to show filename option
                    custom_filename_input = gr.Textbox(label="Custom Filename Prefix (Optional)", placeholder="my_creation_prefix")
                    if allow_model_change:
                        new_model_id_input = gr.Textbox(label="Change Model ID (e.g., stabilityai/sdxl-base-1.0)", value=initial_model_id)
                        change_model_button = gr.Button("Load New Model")


            # --- Right Column: Output ---
            with gr.Column(scale=3, min_width=520): # Output column
                gr.Markdown("## 🖼️ Generated Image")
                image_output = gr.Image(label="Output Image", type="pil", height=512, show_label=False, show_download_button=True, visible=False)
                # A separate download button is not strictly needed here if gr.Image's own button is used.
                # download_button_separate = gr.DownloadButton(label="💾 Download Image with Custom Name", visible=False)

                info_output = gr.Textbox(label="Generation Info", lines=4, interactive=False) # Increased lines for more info

        # --- Event Handlers ---
        def on_generate_click_wrapper(prompt, neg_prompt, style, steps, cfg, seed, filename_prefix_val, progress=gr.Progress(track_ τότε=True)):
            # Initial UI updates for loading state
            yield {
                generate_button: gr.update(interactive=False, value="⏳ Generating..."),
                status_textbox: gr.update(value="⏳ Generating image..."),
                image_output: gr.update(visible=False, value=None), # Clear previous image
                # download_button_separate: gr.update(visible=False),
                info_output: gr.update(value=""),
            }

            # Call the actual generation function
            img, info_text_from_generate, dl_btn_update_placeholder = generate_image_fn(
                prompt, neg_prompt, style, steps, cfg, seed,
                custom_filename_input=filename_prefix_val, # Pass the value from the textbox
                progress=progress
            )

            # final_dl_btn_update = dl_btn_update_placeholder # Use the update from generate_image_fn
            # If img is generated, and we used a separate download button, we'd save img to a temp file
            # and set final_dl_btn_update = gr.DownloadButton.update(value=temp_file_path, visible=True)
            # Since gr.Image handles its own download, this separate button is less critical for app.py

            # Final UI updates with results
            yield {
                image_output: gr.update(value=img, visible=True if img else False),
                info_output: gr.update(value=info_text_from_generate),
                generate_button: gr.update(interactive=True, value="🖼️ Generate Image"),
                status_textbox: gr.update(value="✅ Generation complete." if img else "❌ Generation failed."),
                # download_button_separate: final_dl_btn_update
            }

        generate_button.click(
            fn=on_generate_click_wrapper,
            inputs=[prompt_input, negative_prompt_input, style_dropdown, inference_steps_slider, cfg_scale_slider, seed_input, custom_filename_input], # Added custom_filename_input
            outputs=[generate_button, status_textbox, image_output, info_output], # download_button_separate output removed as gr.Image handles its own
        )

        random_seed_button.click(fn=get_random_seed, inputs=None, outputs=seed_input)

        if allow_model_change and 'change_model_button' in locals(): # Check if button was created
            def handle_change_model(new_model_id_str):
                global pipe, current_model_id # Make sure globals are modified
                if new_model_id_str != current_model_id:
                    # Update UI to show loading
                    yield {status_textbox: gr.update(value=f"Unloading current model '{current_model_id}'..."), change_model_button: gr.update(interactive=False), generate_button: gr.update(interactive=False)}

                    pipe = None # Release VRAM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    yield {status_textbox: gr.update(value=f"Loading new model: {new_model_id_str}... Please wait.")}

                    try:
                        # Call load_model directly, not as a generator here for simplicity in this handler
                        # The status_update_fn for load_model can print to console.
                        load_model(new_model_id_str, status_update_fn=lambda msg: print(f"[Model Load]: {msg}"))

                        # Update the main markdown title (this is a bit of a hack for Gradio Blocks)
                        # It's better to have a dedicated gr.Markdown component for the title if dynamic updates are frequent.
                        # For now, the status_textbox confirms the load.
                        # The initial_model_id in the Markdown is static.

                        yield {
                            status_textbox: gr.update(value=f"Model '{current_model_id}' loaded successfully."), # current_model_id is updated by load_model
                            change_model_button: gr.update(interactive=True),
                            generate_button: gr.update(interactive=True)
                        }
                    except Exception as e:
                        yield {
                            status_textbox: gr.update(value=f"Failed to load {new_model_id_str}: {e}"),
                            change_model_button: gr.update(interactive=True),
                            generate_button: gr.update(interactive=True) # Re-enable generate even if model load fails, user might try again
                        }
                else:
                    yield {status_textbox: gr.update(value=f"Model '{new_model_id_str}' is already loaded.")}

            change_model_button.click(
                handle_change_model,
                inputs=[new_model_id_input],
                outputs=[status_textbox, change_model_button, generate_button] # Components to update
            )
    return demo

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gradio app for Stable Diffusion with advanced UI.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (slow)")
    parser.add_argument("--no_float16", action="store_true", help="Disable float16 (use float32)")
    parser.add_argument("--attention_slicing", action="store_true", help="Enable attention slicing (saves VRAM)")
    parser.add_argument("--allow_model_change", action="store_true", help="Allow changing model via UI (experimental)")

    args = parser.parse_args()

    print("--- Configuration ---")
    print(f"Model ID: {args.model_id}")
    print(f"Device: {'CPU' if args.cpu else 'CUDA (if available)'}")

    use_fp16 = True
    if args.no_float16 or args.cpu or not torch.cuda.is_available():
        use_fp16 = False
    print(f"Use float16: {use_fp16}")
    print(f"Attention Slicing: {args.attention_slicing}")
    print(f"Allow Model Change in UI: {args.allow_model_change}")
    print("---------------------")

    print(f"Attempting initial load of model: {args.model_id}")
    try:
        load_model(
            model_id_to_load=args.model_id,
            use_float16=use_fp16,
            use_attention_slicing=args.attention_slicing,
            status_update_fn=print
        )
    except Exception as e:
        print(f"FATAL: Could not load the initial model '{args.model_id}'. Error: {e}")
        # exit(1) # Optionally exit

    gradio_ui = create_gradio_interface(initial_model_id=current_model_id or args.model_id, allow_model_change=args.allow_model_change)

    print(f"Launching Gradio app on port {args.port}...")
    gradio_ui.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, debug=True)

print("app.py script finished.")
