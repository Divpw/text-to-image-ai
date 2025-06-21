import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
import random
import argparse

# --- Model and Pipeline Setup ---
pipe = None
model_id = "stabilityai/stable-diffusion-2-1-base" # Default model

def load_model(model_id_to_load=model_id, use_float16=True, use_attention_slicing=False):
    """Loads the Stable Diffusion model."""
    global pipe
    if pipe is None:
        print(f"Loading model: {model_id_to_load}...")
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id_to_load, subfolder="scheduler")

        pipeline_args = {
            "scheduler": scheduler,
        }
        if use_float16 and torch.cuda.is_available():
            pipeline_args["torch_dtype"] = torch.float16

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id_to_load,
                **pipeline_args
            )

            if torch.cuda.is_available():
                print("Moving model to CUDA.")
                pipe = pipe.to("cuda")
            else:
                print("CUDA not available. Running on CPU (this will be very slow).")

            if use_attention_slicing:
                print("Enabling attention slicing.")
                pipe.enable_attention_slicing()

            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            pipe = None # Ensure pipe is None if loading failed
            raise # Re-raise the exception to notify the caller
    else:
        print("Model already loaded.")

def generate_image(prompt, negative_prompt, seed_value, num_inference_steps=25, guidance_scale=7.5):
    """Generates an image based on the prompt, negative prompt, and seed."""
    if pipe is None:
        # This should ideally not be reached if pre-loading is enforced
        return None, "Model not loaded. Please start the application correctly."

    print(f"Generating image with prompt: '{prompt}'")
    if negative_prompt:
        print(f"Negative prompt: '{negative_prompt}'")

    try:
        seed = int(seed_value)
    except (ValueError, TypeError):
        seed = random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)

    try:
        # For CPU, autocast is not typically used, and float16 might not be supported well.
        if device == "cuda" and pipe.torch_dtype == torch.float16:
            with torch.autocast("cuda"):
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator
                ).images[0]
        else: # CPU or non-float16 execution
             image = pipe(
                prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator
            ).images[0]
        print("Image generated.")
        return image, f"Seed used: {seed}"
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None, f"Error: {e}"

# --- Gradio Interface Definition ---
def create_gradio_interface():
    with gr.Blocks(css="footer {display: none !important}", title="Stable Diffusion Image Generator") as demo:
        gr.Markdown("# Stable Diffusion Image Generator")
        gr.Markdown("Enter a prompt and (optionally) a negative prompt, then click 'Generate Image'.")

        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., A photo of an astronaut riding a horse on the moon", lines=2)
                negative_prompt_input = gr.Textbox(label="Negative Prompt (Optional)", placeholder="e.g., blurry, low quality, ugly, text, watermark, disfigured", lines=2)
                seed_input = gr.Textbox(label="Seed (Optional)", placeholder="Enter a number or leave blank for random")
                with gr.Row():
                    num_steps_slider = gr.Slider(minimum=10, maximum=100, value=25, step=1, label="Number of Inference Steps")
                    guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.1, label="Guidance Scale")
                generate_button = gr.Button("Generate Image", variant="primary")
            with gr.Column(scale=2):
                image_output = gr.Image(label="Generated Image", type="pil", show_download_button=True, height=512) # Adjust height as needed
                status_output = gr.Textbox(label="Status", interactive=False)

        generate_button.click(
            generate_image,
            inputs=[prompt_input, negative_prompt_input, seed_input, num_steps_slider, guidance_slider],
            outputs=[image_output, status_output]
        )

        gr.Markdown("## Tips for Better Prompts:\n"
                    "- Be specific: 'A hyperrealistic 4K photo of a red apple on a wooden table' is better than 'apple'.\n"
                    "- Use artistic styles: '...in the style of Van Gogh', '...as a watercolor painting', '...cyberpunk art'.\n"
                    "- Add details: '...with dramatic lighting', '...detailed fur', '...wearing a tiny hat'.\n"
                    "- Use negative prompts to exclude unwanted elements.")

        gr.Markdown("--- \n *Model: stabilityai/stable-diffusion-2-1-base (or as specified by --model_id)*")
    return demo

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gradio app for Stable Diffusion.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=model_id,
        help=f"HuggingFace model ID to use (default: {model_id})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the Gradio app on (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing to create a public link"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (overrides CUDA detection for float16)"
    )
    parser.add_argument(
        "--no_float16",
        action="store_true",
        help="Disable float16 precision (use float32). Useful if float16 causes issues."
    )
    parser.add_argument(
        "--attention_slicing",
        action="store_true",
        help="Enable attention slicing for lower VRAM usage (can be slightly slower)."
    )

    args = parser.parse_args()

    print("--- Configuration ---")
    print(f"Model ID: {args.model_id}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print(f"Force CPU: {args.cpu}")

    use_fp16 = True
    if args.no_float16 or args.cpu or not torch.cuda.is_available():
        use_fp16 = False
    print(f"Use float16: {use_fp16}")
    print(f"Enable Attention Slicing: {args.attention_slicing}")
    print("---------------------")

    try:
        load_model(
            model_id_to_load=args.model_id,
            use_float16=use_fp16,
            use_attention_slicing=args.attention_slicing
        )
    except Exception as e:
        print(f"Failed to load the model: {e}")
        print("Exiting application.")
        exit(1)

    if pipe is None:
        print("Model could not be loaded. The Gradio application cannot start.")
    else:
        print("Launching Gradio interface...")
        app_ui = create_gradio_interface()
        app_ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, debug=True)
