import runpod
import os
import tempfile
import shutil
import json
import requests # For uploading the result
import uuid
import time
# import time # For adding delays in retry mechanism - No longer needed
from predict import Predictor, download_weights # Assuming Predictor can be imported
import yt_dlp # For downloading video from URL

# Initialize the Predictor globally
# This will run the setup() method once when the worker starts.
print("Initializing Predictor...")
predictor = Predictor()
predictor.setup() # Explicitly call setup
print("Predictor initialized.")

# upload_to_fileio and upload_to_tempsh functions have been removed.

def generate_unique_filename(original_filename=None):
    """
    Generates a unique filename using UUID and timestamp.
    
    Args:
        original_filename (str, optional): Original filename to extract extension from
        
    Returns:
        str: Unique filename with .mp4 extension
    """
    # Generate unique identifier using UUID and timestamp
    unique_id = str(uuid.uuid4())[:8]  # First 8 characters of UUID
    timestamp = str(int(time.time()))  # Unix timestamp
    
    # Extract extension from original filename if provided, otherwise default to .mp4
    if original_filename:
        _, ext = os.path.splitext(original_filename)
        if not ext:
            ext = '.mp4'
    else:
        ext = '.mp4'
    
    # Create unique filename: restored_video_{timestamp}_{uuid}.mp4
    unique_filename = f"restored_video_{timestamp}_{unique_id}{ext}"
    return unique_filename

def upload_to_bunnycdn(file_path, file_name):
    """
    Uploads a file to BunnyCDN storage and returns the public URL.

    Args:
        file_path (str): The local path to the file to be uploaded.
        file_name (str): The desired name of the file in BunnyCDN storage.

    Returns:
        str: The public URL of the uploaded file if successful, None otherwise.
    """
    storage_zone_name = "zockto" # As per the URL structure
    storage_path_prefix = "videos" # As per the URL structure
    access_key = "17e23633-2a7a-4d29-9450be4d6c8e-e01f-45f4"
    
    upload_url = f"https://storage.bunnycdn.com/{storage_zone_name}/{storage_path_prefix}/{file_name}"
    
    headers = {
        "AccessKey": access_key,
        "Content-Type": "video/mp4",
    }

    try:
        with open(file_path, 'rb') as f_data:
            print(f"Attempting to upload {file_name} to BunnyCDN at {upload_url}...")
            response = requests.put(upload_url, headers=headers, data=f_data)
        
        if response.status_code == 201:
            public_url_base = "https://zockto.b-cdn.net" # As per the public URL structure
            public_url = f"{public_url_base}/{storage_path_prefix}/{file_name}"
            print(f"File {file_name} uploaded successfully to BunnyCDN: {public_url}")
            return public_url
        else:
            print(f"BunnyCDN upload failed for {file_name}: Status {response.status_code} - Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error uploading {file_name} to BunnyCDN: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path} for BunnyCDN upload.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during BunnyCDN upload of {file_name}: {e}")
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

            # Generate unique filename instead of using the original filename
            original_filename = os.path.basename(output_video_path_str)
            unique_filename = generate_unique_filename(original_filename)
            print(f"Generated unique filename: {unique_filename}")

            bunnycdn_url = upload_to_bunnycdn(output_video_path_str, unique_filename)

            if bunnycdn_url:
                print(f"Successfully uploaded to BunnyCDN: {bunnycdn_url}")
                return {
                    "message": "Prediction successful. Output video uploaded to BunnyCDN.",
                    "output_video_url": bunnycdn_url,
                    "filename": unique_filename
                }
            else:
                print(f"Prediction successful, but BunnyCDN upload failed for {output_video_path_str}.")
                return {
                    "error": "Prediction successful, but failed to upload output video to BunnyCDN."
                }
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
    # Test code related to upload_to_fileio and upload_to_tempsh has been removed.
    # If you run this script directly (python handler.py), it will attempt to start the RunPod serverless worker.
    # Example usage (requires models to be downloaded, etc.):
    # test_event = {
    #     "input": {
    #         "video_url": "https://www.youtube.com/watch?v=your_video_id",
    #         # Add other params if needed
    #     }
    # }
    # print("Simulating a handler event locally (mocking may be needed for full test):")
    # result = handler(test_event) # This would require mocking predictor and yt_dlp if run
    # print(f"Handler result: {result}")
    pass

# Start the serverless worker
runpod.serverless.start({"handler": handler})
