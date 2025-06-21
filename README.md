# Enhanced Text-to-Image AI with Stable Diffusion

This project provides a comprehensive system to generate high-quality images from text prompts using Stable Diffusion. It features a user-friendly Gradio interface runnable on Google Colab (free GPU) or locally, and an optional FastAPI backend for programmatic access.

## ✨ Key Features

-   **Advanced Gradio UI:**
    -   Modern 2-column layout: Inputs on the left, image output on the right.
    -   **Prompt Controls:** Main prompt, negative prompt.
    -   **Style Presets:** Dropdown for styles like Realistic, Anime, Fantasy Art, Digital Painting, 3D Render.
    -   **Generation Parameters:** Sliders for Inference Steps (10-100) and CFG Scale (1.0-20.0).
    -   **Seed Control:** Input field for seed, with a "Randomize Seed" button.
    -   **Loading Indicators:** Visual feedback during image generation.
-   **Output Management:**
    -   Displays generated image with timestamp, seed, model info, and suggested filename.
    -   Download button for the generated image.
    -   **Custom Filename Prefix:** Option to specify a prefix for downloaded/saved files.
    -   **(Colab Only) Save to Google Drive:** Option to automatically save generated images to your Google Drive.
-   **Backend Flexibility:**
    -   **`main.ipynb`:** Google Colab notebook for easy setup and execution on free GPUs. Generates a public shareable link.
    -   **`app.py`:** Python script for running the Gradio application locally, supporting command-line arguments for configuration.
    -   **`api.py` (Optional):** FastAPI server providing a POST endpoint (`/generate/`) for programmatic image generation and a GET endpoint (`/health/`) for status checks.
-   **Stable Diffusion Integration:**
    -   Utilizes HuggingFace `diffusers` library.
    -   Defaults to `runwayml/stable-diffusion-v1-5` for compatibility with free Colab T4 GPUs.
    -   Supports changing to other models (including SDXL, VRAM permitting) by modifying the model ID in `main.ipynb` (Cell 4), `app.py` (via CLI argument `--model_id` or experimental UI feature), and `api.py` (via `API_DEFAULT_MODEL_ID` env var).
    -   No paid Hugging Face tokens required for default operation.

## 📂 Project Structure

-   `main.ipynb`: The primary Google Colab notebook with all UI enhancements and Google Drive integration.
-   `app.py`: Python script for running the Gradio UI locally.
-   `api.py`: Optional FastAPI server for programmatic image generation.
-   `README.md`: This file, providing an overview, setup, and usage instructions.

## 🚀 Setup and Usage

### 1. Google Colab (`main.ipynb`) - Recommended for easy start

1.  **Open in Colab:**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/your-repo-name/blob/main/main.ipynb)  *(Replace with your actual GitHub repo link after creation if this project is hosted on GitHub)*
    *   Alternatively, go to [Google Colab](https://colab.research.google.com/), select `File` -> `Upload notebook...`, and upload the `main.ipynb` file from this repository.

2.  **Enable GPU:**
    *   In Colab: `Runtime` -> `Change runtime type` -> Select `GPU` (e.g., T4) from the "Hardware accelerator" dropdown. Click `Save`.

3.  **Run the Cells:**
    *   Execute cells in `main.ipynb` sequentially by clicking the "play" button on each cell or using `Shift+Enter`.
    *   **Cell 1:** Installs necessary Python libraries. This may take a few minutes.
    *   **Cell 2:** Imports libraries and defines helper functions, style presets, Google Drive functions, and the core image generation logic.
    *   **Cell 3:** Defines the Gradio User Interface structure.
    *   **Cell 4:** Loads the chosen Stable Diffusion model. The default is `runwayml/stable-diffusion-v1-5`. You can edit this cell to choose a different model ID. Model downloading and loading can take several minutes, especially on the first run for a new model.
    *   **Cell 5:** Launches the Gradio interface. Wait for a public URL (usually ending in `gradio.live` or `gradio.app`) to appear in the cell output. Click this URL to open the UI in a new browser tab.

4.  **Using the Interface (`main.ipynb`):**
    *   The UI is organized into a 2-column layout.
    *   **Left Column (Inputs):**
        *   Enter your main **Prompt** and an optional **Negative Prompt**.
        *   Select an **Artistic Style Preset** from the dropdown.
        *   Adjust **Inference Steps** and **CFG Scale (Guidance)** using the sliders.
        *   Enter a specific **Seed** number or click "🎲 Randomize Seed" for a random one.
        *   Under "Output Options":
            *   Enter a **Custom Filename Prefix** if desired.
            *   Check **"Save to Google Drive"** if you want to save the image to your Google Drive. The first time you use this in a session, Colab will ask for permission to access your Drive. Images are saved to `My Drive/AI_Generated_Images/StableDiffusion/`.
    *   Click the **"🖼️ Generate Image"** button.
    *   **Right Column (Output):**
        *   The generated image will appear here.
        *   A "💾 Download Image" button will become active, allowing you to download the image. The filename will include your custom prefix (if any), model name, seed, and timestamp.
        *   "Generation Info & Logs" will display details about the generation process, including the seed used, timestamp, model ID, style, final filename, and any messages related to Google Drive saving.
    *   The "Status" bar at the top provides feedback on model loading and generation progress.

### 2. Local Gradio App (`app.py`)

For running the Gradio interface on your local machine (requires a suitable Python environment and preferably an NVIDIA GPU with CUDA).

1.  **Prerequisites:**
    *   Python 3.8+
    *   Git (for cloning the repository)
    *   NVIDIA GPU with CUDA and cuDNN installed (for GPU acceleration). CPU-only mode is supported but very slow.

2.  **Clone the repository (if you haven't already):**
    ```bash
    # git clone https://github.com/yourusername/your-repo-name.git
    # cd your-repo-name
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

4.  **Install dependencies:**
    ```bash
    # Ensure PyTorch is installed with CUDA support if applicable.
    # Visit https://pytorch.org/get-started/locally/ for the correct pip/conda command for your system.
    # Example for CUDA 11.8:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # For other CUDA versions (e.g., 12.1), replace cu118 with cu121.

    # Install other dependencies:
    pip install diffusers transformers accelerate gradio Pillow bitsandbytes
    ```
    *   `bitsandbytes` is optional but recommended for potential 8-bit optimizations.

5.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
    Or using the Gradio CLI:
    ```bash
    gradio app.py
    ```
    The `app.py` script supports several command-line arguments for configuration. Use `python app.py --help` to see all options:
    *   `--model_id`: Specify the HuggingFace model ID (e.g., `"stabilityai/stable-diffusion-2-1-base"`).
    *   `--port`: Set the port for the Gradio app (default: 7860).
    *   `--share`: Create a public Gradio link (useful for sharing access temporarily).
    *   `--cpu`: Force CPU usage (generation will be very slow).
    *   `--no_float16`: Disable float16 precision (uses float32, may increase VRAM usage but can help with some GPUs/models).
    *   `--attention_slicing`: Enable attention slicing to reduce VRAM usage (may slightly slow down generation).
    *   `--allow_model_change`: (Experimental) Adds a UI option to change the loaded model ID dynamically.

    Once started, access the UI by opening the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser.

### 3. API Usage (`api.py`) - Optional

For programmatic image generation, a FastAPI server is included.

1.  **Install API dependencies:**
    (Ensure base dependencies like `torch`, `diffusers`, `Pillow` from the Gradio app setup are also installed in the same environment.)
    ```bash
    pip install fastapi uvicorn[standard] python-multipart
    ```

2.  **Run the API server:**
    From the repository root directory:
    ```bash
    python api.py
    ```
    Or with Uvicorn for more control (recommended for development/production):
    ```bash
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
    ```
    The API loads the model specified by the `API_DEFAULT_MODEL_ID` environment variable (defaults to `runwayml/stable-diffusion-v1-5`) on startup.

3.  **API Endpoints:**
    *   **`POST /generate/`**: Generates an image.
        *   **Request Body (JSON):**
            ```json
            {
                "prompt": "A serene bioluminescent forest at night, mystical creatures",
                "negative_prompt": "daytime, city, people, harsh lighting",
                "style_name": "Fantasy Art", // Valid style from STYLE_PRESETS or "None"
                "num_inference_steps": 30,   // e.g., 10-100
                "guidance_scale": 7.0,       // e.g., 1.0-20.0
                "seed": 98765                // Optional; random if omitted or < 0
            }
            ```
        *   **Success Response:** `200 OK` with the generated image as `image/png`.
        *   **Error Responses:** `400 Bad Request` (e.g., invalid `style_name`), `503 Service Unavailable` (model not loaded), `500 Internal Server Error` (generation issues).
    *   **`GET /health/`**: Health check. Returns JSON with API status, loaded model info, and system details.

4.  **Environment Variables for `api.py`:**
    *   `API_DEFAULT_MODEL_ID`: Model ID for API startup (e.g., `"stabilityai/sdxl-base-1.0"`).
    *   `API_USE_FLOAT16`: `true` (default) or `false` for float16 precision.
    *   `API_ATTENTION_SLICING`: `true` (default) or `false` for attention slicing.
    *   `API_HOST`: Host for Uvicorn (default: `0.0.0.0`).
    *   `API_PORT`: Port for Uvicorn (default: `8000`).
    *   `API_RELOAD`: `true` (default) or `false` for Uvicorn's auto-reload feature.

    Example of running with an environment variable:
    ```bash
    API_DEFAULT_MODEL_ID="stabilityai/stable-diffusion-2-1-base" python api.py
    ```

## ⚙️ Model Configuration

-   **Default Model:** `runwayml/stable-diffusion-v1-5` is the default for its balance of quality and resource efficiency, making it suitable for free Colab T4 GPUs.
-   **Changing Models:** You can use other models from Hugging Face Hub.
    -   **Colab (`main.ipynb`):** Edit the `MODEL_TO_LOAD` variable in Cell 4.
    -   **Local App (`app.py`):** Use the `--model_id` command-line argument. The experimental UI option for model changing (if enabled with `--allow_model_change`) also allows this.
    -   **API (`api.py`):** Set the `API_DEFAULT_MODEL_ID` environment variable before starting the server.
-   **SDXL Models:** Models like `stabilityai/sdxl-base-1.0` generally produce higher-quality images but require significantly more VRAM (typically >12-16GB). They may not run on standard free Colab T4 GPUs or systems with limited GPU memory. Ensure your hardware is sufficient if you choose to use SDXL models.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please feel free to fork the repository, make your changes, and submit a pull request. If you encounter any problems or have suggestions, please open an issue on GitHub.

## 📜 License

This project is open-sourced under the MIT License. (If a `LICENSE` file is not present in the repository, consider adding one with the standard MIT License text.)
