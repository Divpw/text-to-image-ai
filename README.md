# Text-to-Image Generation with Stable Diffusion on Google Colab

This project provides a simple way to generate images from text prompts using Stable Diffusion, hosted on a free Google Colab GPU. It uses the HuggingFace `diffusers` library and provides a Gradio interface for easy interaction.

## Features

-   **Stable Diffusion:** Leverages powerful pre-trained Stable Diffusion models for high-quality image generation.
-   **HuggingFace Diffusers:** Utilizes the `diffusers` library for easy model loading and pipeline management.
-   **Google Colab:** Runs on a free GPU instance in Google Colab, making it accessible to everyone.
-   **Gradio UI:** Offers a user-friendly web interface with:
    -   Textbox for your main prompt.
    -   Textbox for a negative prompt (to specify what you *don't* want to see).
    -   Optional seed input for reproducible results.
    -   "Generate" button to create the image.
    -   Image display area with a download option for the generated image.

## Project Structure

-   `main.ipynb`: The main Google Colab notebook. This is where you'll run the code to install dependencies, load the model, and launch the Gradio UI.
-   `app.py`: An optional Python script that contains the Gradio app. This can be used to run the application locally if you have a suitable environment with a GPU.
-   `README.md`: This file, providing an overview and setup instructions.

## Setup and Usage (Google Colab - `main.ipynb`)

1.  **Open in Colab:**
    *   Go to [Google Colab](https://colab.research.google.com/).
    *   Click on `File` -> `Upload notebook...`
    *   Upload the `main.ipynb` file from this repository.
    *   Alternatively, if this repository is public on GitHub, you can open it directly by replacing `github.com` with `colab.research.google.com/github` in the repository URL and navigating to `main.ipynb`.

2.  **Enable GPU:**
    *   In the Colab notebook, go to `Runtime` -> `Change runtime type`.
    *   Select `GPU` from the `Hardware accelerator` dropdown menu (e.g., T4 GPU).
    *   Click `Save`.

3.  **Run the Cells:**
    *   Execute the cells in the `main.ipynb` notebook one by one.
    *   The first few cells will install the necessary libraries (like `diffusers`, `transformers`, `accelerate`, and `gradio`). This might take a few minutes.
    *   The subsequent cells will load the Stable Diffusion model. This will also take some time, especially the first time it downloads the model weights.
    *   The final cell will launch the Gradio interface. You'll see a public URL (usually ending with `gradio.live`) in the output. Click this URL to open the UI in a new browser tab.

4.  **Generate Images:**
    *   In the Gradio interface:
        *   Enter your desired **prompt** (e.g., "A photo of an astronaut riding a horse on the moon").
        *   Optionally, enter a **negative prompt** (e.g., "blurry, low quality, ugly, text, watermark").
        *   Optionally, set a **seed** (an integer) if you want to try and reproduce a specific image later. Leave it blank for random results.
        *   Click the **"Generate Image"** button.
        *   Wait for the image to be generated. It will appear in the output area.
        *   You can download the image using the download icon/button provided by Gradio.

## Local Usage (`app.py`) - Optional

If you have a local machine with a compatible GPU (usually NVIDIA) and Python environment set up:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust cuXXX for your CUDA version
    pip install diffusers transformers accelerate gradio
    ```
    *Note: Ensure you have CUDA installed if using an NVIDIA GPU.*

4.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
    This will start a local Gradio server, and you can access the UI by opening the provided URL (usually `http://127.0.0.1:7860`) in your browser.

## Model Used

This implementation typically uses a pre-trained Stable Diffusion model from Hugging Face Hub, such as `runwayml/stable-diffusion-v1-5` or a more recent/optimized version. The specific model can be adjusted in the code.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more details (if one is added).
