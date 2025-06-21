# Ultra-Professional AI Image Generation System with Stable Diffusion & SDXL

Welcome to an advanced AI image generation system designed for high-quality output and a professional user experience. This project leverages Stable Diffusion (including SDXL models), Real-ESRGAN for upscaling, and offers multiple interfaces: a Google Colab notebook, a local Gradio application, and a FastAPI backend. It's built with a focus on free-tier GPU compatibility (like Colab's T4), reproducibility, and ease of use.

## ✨ Core Features

-   **GitHub-Synced Colab Notebook (`main.ipynb`):**
    -   Automatically clones or pulls the latest code from your specified GitHub repository on startup, ensuring you're always running the most up-to-date version.
    -   Optimized for Google Colab's free tier GPUs (T4, P100).
-   **Advanced Diffusion Model Handling:**
    -   **SDXL Prioritization:** Attempts to load SDXL models (e.g., `stabilityai/sdxl-base-1.0`) for superior image quality.
    -   **SD 1.5 Fallback:** Automatically falls back to Stable Diffusion 1.5 (e.g., `runwayml/stable-diffusion-v1-5`) if SDXL fails (e.g., due to VRAM limits), ensuring functionality on various GPU tiers.
    -   Memory-efficient loading using `torch.float16` and attention slicing where applicable.
-   **Ultra High-Quality Output:**
    -   **Real-ESRGAN Upscaling:** Optional x4 image upscaling using Real-ESRGAN for significantly sharper and more detailed final images.
    -   Selectable image dimensions (e.g., 512x512, 768x768, 1024x1024) to balance quality and generation speed.
-   **Polished Gradio User Interface (`main.ipynb` & `app.py`):**
    -   Clean, responsive 2-column layout (Inputs | Output).
    -   **Comprehensive Prompt Controls:**
        -   Main prompt and negative prompt textboxes.
        -   Curated style presets: "Realistic", "Cyberpunk", "Anime", "Watercolor", "3D Render".
        -   Sliders for Inference Steps (10-50) and CFG Scale (1.0-15.0).
        -   Seed input with a "Randomize Seed" button (-1 for random).
    -   Loading spinners and status messages for a smooth user experience.
    -   Aesthetic styling for a professional look and feel.
-   **Rich Output Features:**
    -   Image preview with detailed generation info: final size, timestamp, seed, model used, style, and logs.
    -   Download button for the final (potentially upscaled) image with a descriptive filename (e.g., `[prompt_prefix]_WxH_[upscaled_]seed_timestamp.png`).
    -   **(Colab Only)** Option to save images directly to your Google Drive.
-   **Flexible Deployment:**
    -   **`app.py`:** Run the Gradio app locally with extensive CLI arguments for customization.
    -   **`api.py` (Optional):** FastAPI server for programmatic image generation via a `POST /generate/` endpoint, supporting all key features including SDXL, dimensions, and upscaling. Includes a `/health/` check.
-   **Free & Open:**
    -   Designed to work entirely on free resources (no paid Hugging Face tokens required for core functionality).
    -   Well-commented code for easy understanding and modification.

## 📂 Project Structure

-   `main.ipynb`: The primary Google Colab notebook. **Start here for the easiest experience.**
-   `app.py`: Python script for running the Gradio UI locally on your machine.
-   `api.py`: Optional FastAPI server for developers looking to integrate image generation into other applications.
-   `README.md`: This file.
-   *(Potentially other utility scripts or a `requirements.txt` if added to the synced GitHub repo).*

## 🚀 Setup and Usage

### 1. Google Colab (`main.ipynb`) - Recommended Path

This is the most straightforward way to use the system, especially with free GPU access.

1.  **Fork the Repository (Crucial First Step if you want to customize):**
    *   If you intend to modify or manage your own version of the `app.py` or other scripts that `main.ipynb` might pull, **fork the original GitHub repository** to your own GitHub account.
    *   The Colab notebook will be configured to pull from a GitHub repo.

2.  **Open `main.ipynb` in Colab:**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/main.ipynb)
        *(**Important:** Replace `YOUR_USERNAME/YOUR_REPO` with the actual path to **your forked repository** or the main project repository if you are just running it.)*
    *   Alternatively, download `main.ipynb` and upload it to Colab (`File` -> `Upload notebook...`).

3.  **Configure GitHub Repository URL in Colab:**
    *   In the **first code cell** of `main.ipynb` (titled "1. Setup Environment & Sync with GitHub Repository"), **update the `GITHUB_REPO_URL` variable** to point to your repository URL.
    *   Example: `GITHUB_REPO_URL = "https://github.com/YourGitHubUsername/YourForkedRepoName.git"`

4.  **Enable GPU Accelerator:**
    *   In Colab: `Runtime` -> `Change runtime type`.
    *   Select `GPU` from the "Hardware accelerator" dropdown (T4 is common for free tier). Click `Save`.

5.  **Run Cells Sequentially:**
    *   Execute each cell in `main.ipynb` from top to bottom by clicking the "play" icon or using `Shift+Enter`.
    *   **Cell 1 (Setup & Sync):** Clones your specified GitHub repo (or pulls updates if already cloned into `/content/AI_Art_Repo`).
    *   **Cell 2 (Install Dependencies):** Installs Python libraries including `diffusers`, `realesrgan`, `gradio`, etc. This can take a few minutes.
    *   **Cell 3 (Imports & Helpers):** Defines core functions, style presets, upscaler logic.
    *   **Cell 4 (Define Gradio UI):** Sets up the structure of the web interface.
    *   **Cell 5 (Load Main Diffusion Model):** Attempts to load SDXL, with fallback to SD1.5. This is a time-consuming step. Monitor console output.
    *   **Cell 6 (Load Upscaler Model):** Loads the Real-ESRGAN model. Also takes time.
    *   **Cell 7 (Launch Gradio UI):** Starts the Gradio app and provides a public URL (ending in `gradio.live` or `gradio.app`). Click this link.

6.  **Using the Gradio Interface (Colab):**
    *   **Left Panel (Controls):**
        *   Input your **Prompt** and **Negative Prompt**.
        *   Choose an **Artistic Style**.
        *   Select **Image Dimensions**. Be mindful of VRAM; 1024x1024 with SDXL on a T4 GPU can be slow or cause errors. 512x512 or 512x768 are safer starting points for SDXL on T4. SD1.5 works best at 512x512.
        *   Adjust **Inference Steps** and **CFG Scale**.
        *   Set a **Seed** (-1 for random) or use "🎲 Randomize".
        *   Optionally, provide a **Custom Filename Prefix**.
        *   Toggle **"✨ Upscale Image (Real-ESRGAN x4)"**.
        *   Toggle **"💾 Save to Google Drive"** (prompts for Drive access if needed). Images save to `My Drive/AI_Generated_Images/UltraProfessionalSD/`.
    *   Click **"🖼️ Generate Image"**.
    *   **Right Panel (Output):**
        *   View the generated (and potentially upscaled) image.
        *   Use the **"📥 Download Image"** button.
        *   Check **"Generation Info & Logs"** for details.
    *   The **Status Bar** at the top provides feedback during model loading and image processing.

### 2. Local Gradio App (`app.py`)

Run the advanced Gradio UI on your local machine.

1.  **Prerequisites:** Python 3.8+, Git, CUDA/cuDNN for NVIDIA GPUs.
2.  **Clone the Repository:** `git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git` (use the main project URL or your fork).
3.  **Virtual Environment & Dependencies:**
    ```bash
    cd YOUR_REPO
    python -m venv venv
    source venv/bin/activate  # Linux/macOS OR venv\Scripts\activate # Windows

    # Install PyTorch with CUDA (see https://pytorch.org/get-started/locally/)
    # Example for CUDA 11.8:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Install other dependencies
    pip install diffusers transformers accelerate gradio Pillow bitsandbytes basicsr realesrgan
    ```
4.  **Run the App:**
    ```bash
    python app.py
    ```
    Or with Gradio CLI: `gradio app.py`
    Use `python app.py --help` for command-line options (e.g., `--model_id`, `--port`, `--no-default_upscale`).

### 3. FastAPI Server (`api.py`) - Optional

For programmatic image generation.

1.  **Install API Dependencies:**
    ```bash
    pip install fastapi uvicorn[standard] python-multipart Pillow
    ```
    (Ensure base dependencies from `app.py` setup are also installed.)
2.  **Run API Server:**
    ```bash
    python api.py
    ```
    Or with Uvicorn: `uvicorn api:app --reload --host 0.0.0.0 --port 8000`
3.  **Endpoints:**
    *   `GET /health/`: Status of API and models.
    *   `POST /generate/`: Generate image. Request body example:
        ```json
        {
            "prompt": "ethereal jellyfish floating in a nebula, astrophotography",
            "negative_prompt": "blurry, noise, text",
            "style_name": "Realistic",
            "dimensions_str": "1024x1024",
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "seed": -1,
            "upscale_active": true
        }
        ```
4.  **Environment Variables for `api.py` (see `api.py` comments for more):**
    *   `API_SDXL_MODEL_ID`, `API_SD15_MODEL_ID`: Set preferred SDXL and fallback SD1.5 models.
    *   `API_USE_FLOAT16`, `API_ATTENTION_SLICING`.
    *   `API_HOST`, `API_PORT`, `API_RELOAD`.

## ⚙️ Model & Upscaler Configuration

-   **Diffusion Models:**
    -   The system prioritizes SDXL models (default: `stabilityai/sdxl-base-1.0`) for higher quality.
    -   If an SDXL model fails to load (common on lower-VRAM GPUs like Colab's T4), it automatically falls back to Stable Diffusion 1.5 (default: `runwayml/stable-diffusion-v1-5`).
    -   You can configure the preferred SDXL and fallback SD1.5 model IDs:
        -   **Colab (`main.ipynb`):** Edit variables in Cell 4 (Load Main Diffusion Model).
        -   **Local App (`app.py`):** Use `--model_id` (this becomes the preferred model). The fallback is hardcoded in `app.py` but could be made a CLI arg.
        -   **API (`api.py`):** Set `API_SDXL_MODEL_ID` and `API_SD15_MODEL_ID` environment variables.
-   **Real-ESRGAN Upscaler:**
    -   Uses `RealESRGAN_x4plus` by default for general-purpose x4 upscaling.
    -   This is integrated into `main.ipynb` and `app.py` (toggleable via UI) and `api.py` (toggleable via request flag).
    -   Upscaling significantly increases detail but also processing time and memory usage.

## VRAM Considerations

-   **SDXL models** are demanding (typically 12GB+ VRAM for 1024x1024). On a Colab T4 (15GB VRAM), 1024x1024 SDXL generation is possible but can be slow and close to VRAM limits. Upscaling an SDXL 1024x1024 image to 4096x4096 will require substantial VRAM and might fail on a T4.
-   **SD 1.5 models** are much lighter (4-6GB VRAM for 512x512).
-   The system uses `torch.float16` and attention slicing by default to optimize VRAM.

## 🤝 Contributing

Your contributions, issues, and feature ideas are highly welcome! Feel free to fork, modify, and submit pull requests.

## 📜 License

This project is open-sourced under the MIT License. (Please include a `LICENSE` file with MIT License text in your repository if one does not already exist.)
