# Ultra-Professional AI Image Generation System with Stable Diffusion & SDXL

Welcome to an advanced AI image generation system designed for high-quality output and a professional user experience. This project leverages Stable Diffusion (including SDXL models), Real-ESRGAN for upscaling, and offers multiple interfaces: a Google Colab notebook, a local Gradio application, and a FastAPI backend. It's built with a focus on compatibility with free CPU tiers on platforms like Hugging Face Spaces, while also supporting GPU acceleration.

## ✨ Core Features

-   **GitHub-Synced Colab Notebook (`main.ipynb`):**
    -   Automatically clones or pulls the latest code from your specified GitHub repository on startup.
    -   Optimized for Google Colab's free tier GPUs (T4, P100) but also functional on CPU.
-   **Advanced Diffusion Model Handling:**
    -   **Flexible Model Choice:** Defaults to SD 1.5 (`runwayml/stable-diffusion-v1-5`) for broad CPU compatibility (e.g., Hugging Face Spaces free tier).
    -   **SDXL Support:** Users can specify SDXL models (e.g., `stabilityai/sdxl-base-1.0`) via CLI arguments (`app.py`) or environment variables (`api.py`). The system attempts to load the specified model, with robust fallback to SD 1.5 if an SDXL model fails (especially on CPU or low-VRAM GPUs).
    -   Memory-efficient loading using `torch.float16` (on GPU) and attention slicing.
-   **Ultra High-Quality Output:**
    -   **Real-ESRGAN Upscaling:** Optional x4 image upscaling using Real-ESRGAN for significantly sharper and more detailed final images. Defaulted to OFF for CPU compatibility, but toggleable.
    -   Selectable image dimensions (e.g., 512x512, 768x768, 1024x1024) to balance quality and generation speed.
-   **Polished Gradio User Interface (`main.ipynb` & `app.py`):**
    -   Clean, responsive 2-column layout.
    -   Comprehensive prompt controls, style presets, generation parameter sliders, seed control.
    -   Loading spinners and status messages, with clear warnings for CPU mode performance.
-   **Rich Output Features:**
    -   Image preview with detailed generation info (size, timestamp, seed, model, style, logs).
    -   Download button with descriptive filenames.
    -   **(Colab Only)** Option to save images to Google Drive.
-   **Flexible Deployment:**
    -   **`app.py`:** Primary application for Hugging Face Spaces deployment. Runs the Gradio UI with CLI arguments for configuration.
    -   **`main.ipynb`:** For development, experimentation, and GPU usage on Google Colab.
    -   **`api.py` (Optional):** FastAPI server for programmatic image generation.
-   **Free & Open:**
    -   Designed to work on free CPU tiers (Hugging Face Spaces) and free GPU tiers (Colab).
    -   No paid Hugging Face tokens required for core functionality with default models.
    -   Well-commented code.

## 📂 Project Structure

-   `app.py`: **Main application file for Hugging Face Spaces deployment.**
-   `requirements.txt`: Lists all necessary Python dependencies.
-   `main.ipynb`: Google Colab notebook for development and GPU-powered generation.
-   `api.py`: Optional FastAPI server.
-   `README.md`: This file.

## 🚀 Setup and Usage

### 1. Hugging Face Spaces (Recommended for Public Sharing)

Deploy `app.py` to a Hugging Face Space for an easily accessible web application.

1.  **Prepare Your GitHub Repository:**
    *   Ensure `app.py`, `requirements.txt`, and this `README.md` are committed and pushed to your GitHub repository.
    *   *(Optional)* If `app.py` needs to access other local files from your repo (e.g., utility scripts), make sure they are also in the repo.

2.  **Create a New Hugging Face Space:**
    *   Go to [Hugging Face Spaces](https://huggingface.co/new-space).
    *   Choose a **Space name**.
    *   Select **"Gradio"** as the Space SDK.
    *   Choose **"CPU basic - 2 vCPU - 16GB RAM"** for the **free tier**.
        *   **Note:** This tier has no GPU. SDXL models will likely not run or be extremely slow. Upscaling will also be very slow. The app defaults to SD 1.5 and upscaling off for this reason.
        *   For GPU acceleration (recommended for SDXL and faster generation/upscaling), you'll need to select a paid GPU hardware option.
    *   Under "Deploy from GitHub repository", select your GitHub repo and branch (usually `main`).
    *   The **"App file"** should be `app.py`.
    *   Click **"Create Space"**.

3.  **Space Configuration (if needed):**
    *   The Space will attempt to build using `requirements.txt`.
    *   If you need specific Python versions or system libraries, you might need to add a `packages.txt` or configure it via a `Dockerfile` (advanced). For this project, `requirements.txt` should suffice for standard CPU/GPU spaces.

4.  **Using the Deployed Space:**
    *   Once built, your Space will be live at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`.
    *   The UI will load with SD 1.5 by default. Be patient, as CPU generation is slow.
    *   If you've upgraded to a GPU Space, you can try specifying SDXL model IDs using the (experimental) UI model changer in `app.py` if you enabled it, or by forking the Space and modifying `app.py`'s default CLI args if needed.

    **Badge for your GitHub Repo:**
    ```markdown
    [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)
    ```
    *(Replace `YOUR_USERNAME/YOUR_SPACE_NAME` with your actual Space URL)*

### 2. Google Colab (`main.ipynb`) - For Development & GPU Usage

Ideal for experimentation, using powerful GPUs for free, and features like direct Google Drive saving.

1.  **Fork & Configure GitHub URL:**
    *   Fork the main project repository to your GitHub account.
    *   In `main.ipynb` (Cell 1), update `GITHUB_REPO_URL` to **your fork's URL**.
2.  **Open in Colab & Enable GPU:**
    *   Open your forked `main.ipynb` in Colab.
    *   Set `Runtime` -> `Change runtime type` -> `GPU`.
3.  **Run Cells:** Execute all cells sequentially. The notebook will clone/pull your repo, install dependencies, load models (SDXL preferred, SD1.5 fallback), load the upscaler, and launch Gradio with a public link.
4.  **Interface Usage:** Detailed in previous README versions and within the notebook's Markdown cells. Includes controls for dimensions, styles, upscaling, and saving to Drive.

### 3. Local Gradio App (`app.py`)

Run on your local machine, preferably with an NVIDIA GPU.

1.  **Clone & Setup Environment:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    cd YOUR_REPO
    python -m venv venv
    source venv/bin/activate # or venv\Scripts\activate
    pip install -r requirements.txt
    # Ensure PyTorch CUDA version matches your system: https://pytorch.org/get-started/locally/
    ```
2.  **Run:**
    ```bash
    python app.py --model_id "runwayml/stable-diffusion-v1-5" # Starts with SD1.5
    # For SDXL (requires good GPU):
    # python app.py --model_id "stabilityai/sdxl-base-1.0" --default_upscale true
    ```
    Use `python app.py --help` for all options.

### 4. FastAPI Server (`api.py`) - Optional

For programmatic access.

1.  **Install API Dependencies:** `pip install fastapi uvicorn[standard] python-multipart`
2.  **Run:** `python api.py` or `uvicorn api:app --reload --host 0.0.0.0 --port 8000`
3.  **Endpoints:** `GET /health/`, `POST /generate/` (see `api.py` and previous README sections for request details). Configure with environment variables (e.g., `API_SDXL_MODEL_ID`).

## ⚙️ Model & Upscaler Configuration

-   **Diffusion Models for `app.py` (Hugging Face Spaces & Local):**
    -   Defaults to `runwayml/stable-diffusion-v1-5` for CPU compatibility.
    -   Use `--model_id` CLI argument to specify a different model (e.g., `stabilityai/sdxl-base-1.0` if on GPU).
    -   The app attempts to load the specified model. If it's an SDXL model and fails, it then tries the SD1.5 fallback.
-   **Diffusion Models for `main.ipynb` (Colab):**
    -   Defaults to trying SDXL (`stabilityai/sdxl-base-1.0`) first, then SD1.5 (`runwayml/stable-diffusion-v1-5`) as fallback. Configurable in Cell 4.
-   **Real-ESRGAN Upscaler:**
    -   Uses `RealESRGAN_x4plus`. Toggleable in UIs. Default is OFF in `app.py` for CPU performance.

## VRAM & Performance Notes

-   **CPU (Hugging Face Free Tier):**
    -   **Only SD 1.5 models are practical.** Generation will be slow (minutes per image).
    -   **SDXL models will likely fail or be unusably slow.**
    -   **Upscaling on CPU is extremely slow** (can take many minutes). It's defaulted to off.
-   **GPU (Colab T4, Upgraded HF Spaces, Local GPU):**
    -   **SDXL (e.g., `stabilityai/sdxl-base-1.0`):** Recommended for best quality. Requires significant VRAM (12GB+ for 1024x1024). On a T4 (15GB), 1024x1024 is possible but can be slow. Upscaling large SDXL outputs further taxes VRAM.
    -   **SD 1.5 (e.g., `runwayml/stable-diffusion-v1-5`):** Much lighter (4-6GB VRAM for 512x512).
-   The system uses `torch.float16` and attention slicing by default (on GPU) to optimize VRAM.

## 🤝 Contributing

Contributions, issues, and feature ideas are welcome!

## 📜 License

This project is open-sourced under the MIT License.
