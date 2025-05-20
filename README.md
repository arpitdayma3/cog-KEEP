# KEEP: Cog implementation for Video Face Super-Resolution

[![Replicate](https://replicate.com/zsxkib/keep/badge)](https://replicate.com/zsxkib/keep)

This repository contains a Cog container for **KEEP (Kalman-Inspired FEaturE Propagation)**, an advanced model for video face super-resolution. This setup is designed for Cog, emphasizing an API-first approach, and is based on the original [KEEP project](https://github.com/jnjaby/KEEP).

This implementation enhances faces in videos, leveraging the KEEP model along with RealESRGAN for optional background and face upscaling, and Facelib for robust face detection. It's built to run efficiently on NVIDIA GPUs.

**Model links and information:**
*   Original Project Page: [github.com/jnjaby/KEEP](https://github.com/jnjaby/KEEP)
*   Original Paper: Feng, R., Li, C., & Loy, C. C. (2024). *Kalman-Inspired FEaturE Propagation for Video Face Super-Resolution*. Proceedings of the European Conference on Computer Vision (ECCV). (The paper is often available on the [project page](https://github.com/jnjaby/KEEP))
*   The main model used here is **KEEP**.
*   This Cog packaging by: [zsxkib on GitHub](https://github.com/zsxkib) / [@zsakib_ on Twitter](https://twitter.com/zsakib_)

## Prerequisites

*   **Docker**: You'll need Docker to build and run the Cog container. [Install Docker](https://docs.docker.com/get-docker/).
*   **Cog**: Cog is required to build and run this model locally. [Install Cog](https://github.com/replicate/cog#install).
*   **NVIDIA GPU**: You'll need an NVIDIA GPU to run the model. This implementation uses PyTorch and CUDA.

## Run locally: It just works!

Running KEEP with Cog is straightforward. Once you have Cog and Docker set up, you can get started in just a few steps. Cog handles all the heavy lifting, including building the necessary environment and automatically downloading all required model weights (like KEEP, RealESRGAN, and Facelib) the first time you run it.

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/zsxkib/cog-KEEP.git
    cd cog-KEEP
    ```

2.  **Run the model:**
    Simply use the `cog predict` command. You can point to a video on your local system (prefix the path with `@`) or even use a direct public URL. The model weights will be downloaded automatically on the first run, which might take a few minutes. Subsequent runs will be much faster.

    Here's an example using a public URL—it's that easy:
    ```bash
    cog predict \
      -i input_video=@https://replicate.delivery/pbxt/N2eE3jZM0o6JoUfnTXrwCkiklI5BN8USTvVjNk2r3cOvZ4mU/real_1.mp4
    ```

    **More Examples:**

    ```bash
    # Example with background enhancement and face upsampling (using a local file placeholder)
    cog predict \
      -i input_video=@path/to/your_video.mp4 \
      -i bg_enhancement=true \
      -i face_upsample=true \
      -i upscale=2

    # Example using a different face detector and drawing boxes
    cog predict \
      -i input_video=@path/to/your_video.mp4 \
      -i detection_model='yolov5n' \
      -i draw_box=true \
      -i only_center_face=false

    # Example for pre-aligned 512x512 face videos
    cog predict \
      -i input_video=@path/to/aligned_face_video.mp4 \
      -i has_aligned=true \
      -i face_upsample=true
    ```
    Cog will output the path to your restored video, typically in a temporary directory (e.g., `/tmp/cog_outputs/current_prediction/your_video_restored.mp4`).

    For a full list of options, see the "Model parameters" section below or check out `predict.py`.

## How it works

Cog uses `cog.yaml` to set up the Python environment, system packages, and other dependencies. The main logic for video face restoration is in `predict.py`.

*   **`setup()` method**: This is called when the Cog worker starts.
    1.  It sets up cache directories (e.g., `MODEL_CACHE`) for model weights.
    2.  It downloads these pre-trained models:
        *   **KEEP model**: The primary face restoration network (`KEEP-b76feb75.pth`).
        *   **RealESRGAN**: For background enhancement and upsampling restored faces (`RealESRGAN_x2plus.pth`).
        *   **Facelib models**: For face detection (e.g., `detection_Resnet50_Final.pth`, `yolov5n-face.pth`) and parsing (`parsing_parsenet.pth`).
    3.  It initializes `FaceRestoreHelper` for various face processing tasks (detection, alignment, parsing) and `RealESRGANer` for upsampling tasks.
    4.  It loads the KEEP network and other necessary components onto the available device (GPU if present).

*   **`predict()` method**: This handles how the video is processed, based on what you provide.
    1.  It takes inputs like `input_video`, `bg_enhancement`, `detection_model`, and others (see "Model parameters").
    2.  It reads all frames from the input video.
    3.  **Face Detection and Alignment (if `has_aligned` is `false`)**:
        *   It detects face landmarks for each frame using the selected `detection_model`.
        *   It interpolates and smooths these landmarks across frames using a Gaussian filter to ensure temporal consistency and stable alignment.
    4.  **Face Cropping**:
        *   It aligns and crops faces based on the (smoothed) landmarks to a 512x512 resolution.
        *   If `has_aligned` is `true`, it assumes input frames are already 512x512 cropped faces and skips detection and alignment.
        *   It handles frames where no face is detected by using a black placeholder image to maintain video length.
    5.  **Face Restoration with KEEP**:
        *   It processes the batch of cropped faces (normalized tensors) through the KEEP network.
        *   To manage video memory and process long videos, the model processes the video in segments, defined by `max_length`. If a segment contains only a single frame, the frame is duplicated to meet the model's input requirements.
    6.  **Pasting Faces Back**:
        *   It pastes the restored 512x512 faces back into the original video frames.
        *   If `has_aligned` is `false`, it uses inverse affine transformations derived from the smoothed landmarks to correctly position the restored face.
        *   If background enhancement (`bg_enhancement`) is enabled, RealESRGAN processes the full original frame before pasting the restored face. The `upscale` parameter controls the enhancement scale.
        *   If face upsampling (`face_upsample`) is enabled, RealESRGAN upsamples the restored face itself using the `upscale` factor.
        *   The `draw_box` option can be used to draw a bounding box around the pasted face.
    7.  It saves the final processed video as an `mp4` file.

## Model parameters

You can give these parameters to `cog predict` with the `-i` flag (e.g., `-i parameter_name=value`):

| Parameter                    | Description                                                                                                | Default Value        | Type              | Constraints                                                                 |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------------- | :---------------- | :-------------------------------------------------------------------------- |
| `input_video`                | Input video file. (Required)                                                                               | _N/A_                | `Path`            |                                                                             |
| `draw_box`                   | Draw a box around detected/pasted faces in the output video.                                               | `false`              | `bool`            |                                                                             |
| `bg_enhancement`             | Enhance the video background using RealESRGAN.                                                             | `false`              | `bool`            |                                                                             |
| `upscale`                    | Upscaling factor for RealESRGAN (applied if `bg_enhancement` or `face_upsample`