import runpod
import os
import tempfile
import shutil
import json
import requests # For uploading the result
from predict import Predictor, download_weights # Assuming Predictor can be imported
import yt_dlp # For downloading video from URL

# Initialize the Predictor globally
# This will run the setup() method once when the worker starts.
print("Initializing Predictor...")
predictor = Predictor()
predictor.setup() # Explicitly call setup
print("Predictor initialized.")

def upload_to_fileio(file_path):
    """Uploads a file to file.io and returns the public URL."""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post('https://file.io', files={'file': f})
        if response.status_code == 200:
            return response.json().get('link')
        else:
            print(f"File.io upload failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error uploading to file.io: {e}")
        return None

def handler(event):
    job_input = event.get('input', {})

    video_url = job_input.get('video_url')
    if not video_url:
        return {"error": "Missing 'video_url' in input"}

    # Get other parameters from job_input, with defaults matching predict.py if possible
    draw_box = job_input.get('draw_box', False)
    bg_enhancement = job_input.get('bg_enhancement', False)
    upscale = job_input.get('upscale', 1)
    max_length = job_input.get('max_length', 20) # Matches predict.py default
    has_aligned = job_input.get('has_aligned', False)
    only_center_face = job_input.get('only_center_face', True) # Matches predict.py default
    detection_model = job_input.get('detection_model', 'retinaface_resnet50') # Matches predict.py default
    face_upsample = job_input.get('face_upsample', False)
    bg_tile = job_input.get('bg_tile', 400) # Matches predict.py default
    save_video_fps = job_input.get('save_video_fps', None)

    # Create a temporary directory to store the downloaded video
    with tempfile.TemporaryDirectory() as tmpdir:
        local_video_path = os.path.join(tmpdir, 'input_video.mp4')

        # Download video using yt-dlp
        print(f"Downloading video from URL: {video_url}")
        try:
            ydl_opts = {
                'outtmpl': local_video_path,
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # Download best mp4 format
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if not os.path.exists(local_video_path):
                return {"error": f"Failed to download video from {video_url}"}
            print(f"Video downloaded to: {local_video_path}")

        except Exception as e:
            return {"error": f"Failed to download video: {str(e)}"}

        # Run prediction
        # The Predictor.predict method expects a Path object for input_video
        # and other arguments directly.
        # We need to ensure that the Path class used by cog is available or adapt.
        # For simplicity, we'll assume direct path string might work, or we might need
        # to wrap it if predict.py strictly expects cog.Path.
        # Let's try passing the string path first. If it fails, we'll adapt.
        # The output of predict() is a cog.Path object, so we convert it to string.

        print("Starting prediction...")
        try:
            # Simulate cog.Path for the input, as predict.py expects it.
            # A simple mock or using pathlib.Path might be needed if predict.py uses Path methods.
            # For now, let's assume string path is fine, if not, this is where adjustment is needed.
            # The `predict.py` uses `str(input_video)` early on, so a string should be fine.
            output_video_path_obj = predictor.predict(
                input_video=local_video_path, # Pass string directly
                draw_box=draw_box,
                bg_enhancement=bg_enhancement,
                upscale=upscale,
                max_length=max_length,
                has_aligned=has_aligned,
                only_center_face=only_center_face,
                detection_model=detection_model,
                face_upsample=face_upsample,
                bg_tile=bg_tile,
                save_video_fps=save_video_fps
            )
            # The output from predict.py is a cog.Path object, convert to string
            output_video_path_str = str(output_video_path_obj)
            print(f"Prediction output path: {output_video_path_str}")

            if not os.path.exists(output_video_path_str):
                 return {"error": "Prediction finished but output file not found."}

            # Upload the result
            print(f"Uploading {output_video_path_str} to file.io...")
            public_url = upload_to_fileio(output_video_path_str)

            if public_url:
                print(f"File uploaded successfully: {public_url}")
                return {"output_video_url": public_url}
            else:
                return {"error": "Prediction successful, but failed to upload output video."}

        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Prediction failed: {str(e)}"}
        finally:
            # Clean up the temporary output file from predict.py if it's outside tmpdir
            # predict.py saves to /tmp/cog_outputs/current_prediction/
            # This directory is within /tmp so it should be cleaned up by OS eventually,
            # but good to be explicit if possible.
            # However, direct deletion here might be risky if path is not as expected.
            # For now, relying on /tmp cleanup.
            pass


if __name__ == "__main__":
    # This part is for local testing if needed, RunPod calls runpod.serverless.start
    # print("Handler script started directly for testing (not through RunPod).")
    # Example usage (requires models to be downloaded, etc.):
    # test_event = {
    #     "input": {
    #         "video_url": "https://www.youtube.com/watch?v=your_video_id",
    #         # Add other params if needed
    #     }
    # }
    # result = handler(test_event)
    # print(f"Handler result: {result}")
    pass

# Start the serverless worker
runpod.serverless.start({"handler": handler})
