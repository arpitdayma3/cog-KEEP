# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/keep/model_cache/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import subprocess
import shutil
import time
import torch
from typing import Optional, List 
from cog import BasePredictor, Input, Path
import cv2
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from scipy.ndimage import gaussian_filter1d 
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.video_util import VideoReader, VideoWriter
from basicsr.utils.registry import ARCH_REGISTRY



# Constants for model URLs and paths, adapted for Cog
KEEP_MODEL_URL = "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth"
REALESRGAN_MODEL_URL = "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/RealESRGAN_x2plus.pth"
FACELIB_BASE_URL = "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/"
FACELIB_MODELS = [
    "detection_Resnet50_Final.pth",
    "detection_mobilenet0.25_Final.pth",
    "yolov5n-face.pth",
    "yolov5l-face.pth",
    "parsing_parsenet.pth",
]

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.makedirs(MODEL_CACHE, exist_ok=True)

        model_files = [
            "KEEP.tar",
            "facelib.tar",
            "realesrgan.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Copy specific facelib files to weights/facelib
        weights_facelib_dir = "weights/facelib"
        os.makedirs(weights_facelib_dir, exist_ok=True)

        files_to_copy = [
            "detection_Resnet50_Final.pth",
            "parsing_parsenet.pth",
        ]

        for file_name in files_to_copy:
            source_file_path = os.path.join(MODEL_CACHE, "facelib", file_name)
            destination_file_path = os.path.join(weights_facelib_dir, file_name)
            if os.path.exists(source_file_path):
                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied {source_file_path} to {destination_file_path}")
            else:
                print(f"Warning: Source file {source_file_path} not found for copying.")

        self.device = get_device()
        print(f"Using device: {self.device}")

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer
        
        use_half = False
        if torch.cuda.is_available(): 
            no_half_gpu_list = ['1650', '1660'] 
            if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
                use_half = True
        
        realesrgan_model_path = os.path.join(MODEL_CACHE, "realesrgan", "RealESRGAN_x2plus.pth")
        realesrgan_model_dir = os.path.join(MODEL_CACHE, "realesrgan")
        os.makedirs(realesrgan_model_dir, exist_ok=True)
        if not os.path.exists(realesrgan_model_path):
            load_file_from_url(url=REALESRGAN_MODEL_URL, model_dir=realesrgan_model_dir, progress=True, file_name="RealESRGAN_x2plus.pth")

        model_realesrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.bg_upsampler = RealESRGANer(scale=2, model_path=realesrgan_model_path, model=model_realesrgan, tile=400, tile_pad=40, pre_pad=0, half=use_half, device=self.device)
        print("RealESRGANer (bg_upsampler) initialized.")
        self.face_upsampler = self.bg_upsampler 

        keep_model_config = {
            'img_size': 512, 'emb_dim': 256, 'dim_embd': 512, 'n_head': 8, 'n_layers': 9,
            'codebook_size': 1024, 'cft_list': ['16', '32', '64'], 'kalman_attn_head_dim': 48,
            'num_uncertainty_layers': 3, 'cfa_list': ['16', '32'], 'cfa_nhead': 4, 'cfa_dim': 256, 'cond': 1
        }
        self.net = ARCH_REGISTRY.get('KEEP')(**keep_model_config).to(self.device)
        keep_checkpoint_dir = os.path.join(MODEL_CACHE, "KEEP")
        os.makedirs(keep_checkpoint_dir, exist_ok=True)
        keep_ckpt_path = os.path.join(keep_checkpoint_dir, "KEEP-b76feb75.pth")

        if not os.path.exists(keep_ckpt_path):
            load_file_from_url(url=KEEP_MODEL_URL, model_dir=keep_checkpoint_dir, progress=True, file_name="KEEP-b76feb75.pth")
        
        checkpoint = torch.load(keep_ckpt_path, map_location=self.device, weights_only=True) 
        self.net.load_state_dict(checkpoint['params_ema'])
        self.net.eval()
        print("KEEP model loaded.")

        facelib_model_dir = os.path.join(MODEL_CACHE, "facelib")
        os.makedirs(facelib_model_dir, exist_ok=True)
        for model_file in FACELIB_MODELS:
            dest_path = os.path.join(facelib_model_dir, model_file)
            if not os.path.exists(dest_path):
                load_file_from_url(url=FACELIB_BASE_URL + model_file, model_dir=facelib_model_dir, progress=True, file_name=model_file)
        print("Facelib models downloaded.")
        
        self.face_helper = FaceRestoreHelper(
            1, # upscale_factor
            face_size=512, 
            crop_ratio=(1, 1), 
            det_model='retinaface_resnet50', 
            save_ext='png', 
            use_parse=True, 
            device=self.device
        )
        print("FaceRestoreHelper initialized.")

    @staticmethod
    def interpolate_sequence(sequence):
        interpolated_sequence = np.copy(sequence)
        missing_indices = np.isnan(sequence)
        if np.any(missing_indices):
            valid_indices = ~missing_indices
            x = np.arange(len(sequence))
            interpolated_sequence[missing_indices] = np.interp(x[missing_indices], x[valid_indices], sequence[valid_indices])
        return interpolated_sequence

    def _calculate_affine_matrices(self, landmarks_list):
        """Helper method to calculate affine matrices without warping faces.
        This avoids overwriting self.face_helper.cropped_faces when we just need the matrices.
        """
        # Store current affine matrices to restore them later if needed
        original_affine_matrices = self.face_helper.affine_matrices.copy()
        
        # Clear only the affine matrices
        self.face_helper.affine_matrices = []
        
        # Calculate affine matrices for each landmark
        for landmark in landmarks_list:
            affine_matrix = cv2.estimateAffinePartial2D(
                landmark, self.face_helper.face_template, method=cv2.LMEDS)[0]
            self.face_helper.affine_matrices.append(affine_matrix)
        
        return self.face_helper.affine_matrices

    def predict(
        self,
        input_video: Path = Input(description="Input video file"),
        draw_box: bool = Input(description="Draw box around detected faces", default=False),
        bg_enhancement: bool = Input(description="Enable background enhancement using RealESRGAN", default=False),
        upscale: int = Input(description="Upscaling factor for enhancement (RealESRGAN)", default=1, ge=1), 
        max_length: int = Input(description="Maximum number of frames to process in one batch for KEEP model", default=20, ge=1), 
        has_aligned: bool = Input(description="Set to true if faces in the video are already aligned and cropped (512x512)", default=False), 
        only_center_face: bool = Input(description="Process only the center face if multiple faces are detected", default=True), 
        detection_model: str = Input(
            description="Face detection model to use", 
            choices=['retinaface_resnet50', 'retinaface_mobile0.25', 'yolov5n', 'yolov5l'], 
            default='retinaface_resnet50'
        ), 
        face_upsample: bool = Input(description="Upsample restored faces using RealESRGAN", default=False), 
        bg_tile: int = Input(description="Tile size for background upsampler (RealESRGAN)", default=400, ge=64), 
        save_video_fps: Optional[float] = Input(description="FPS for the output video. If None, uses original FPS.", default=None)
    ) -> Path:
        """Process an input video to restore and enhance faces using KEEP model."""
        
        if self.face_helper.det_model != detection_model:
             self.face_helper.det_model = detection_model
             # Re-initialize detector components within FaceRestoreHelper if it caches them based on det_model string
             # For basicsr's FaceRestoreHelper, changing det_model and then calling clean_all() and re-detecting should be enough
             # as it typically initializes detectors on-the-fly.
             print(f"Switched face detection model to: {detection_model}")

        # For cog, output files are typically placed in /tmp or a designated output directory.
        # We'll just create a local dir and Cog will pick up the Path object.

        base_output_dir = "/tmp/cog_outputs"
        os.makedirs(base_output_dir, exist_ok=True)
        output_dir = os.path.join(base_output_dir, "current_prediction")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        
        current_bg_upsampler = self.bg_upsampler if bg_enhancement else None
        current_face_upsampler = self.face_upsampler if face_upsample else None # Assumes self.face_upsampler is RealESRGAN

        if current_bg_upsampler:
            current_bg_upsampler.tile_size = bg_tile 
            print(f"Background upsampling: True, Tile: {bg_tile}, Face upsampling: {face_upsample}")
        else:
            print(f"Background upsampling: False, Face upsampling: {face_upsample}")

        input_img_list = []
        input_video_path_str = str(input_video)
        
        if not os.path.isfile(input_video_path_str):
            raise FileNotFoundError(f"Input video file not found: {input_video_path_str}")

        # basicsr.utils.video_util.VideoReader does not take device in its constructor
        vidreader = VideoReader(input_video_path_str) 
        
        try: 
            img_from_reader = vidreader.get_frame()
            while img_from_reader is not None:
                # VideoReader gives BGR numpy array directly
                input_img_list.append(img_from_reader) 
                img_from_reader = vidreader.get_frame()
            fps = vidreader.get_fps() if save_video_fps is None else save_video_fps
        finally:
            vidreader.close()

        clip_name = os.path.splitext(os.path.basename(input_video_path_str))[0]

        if not input_img_list: # Changed from len(input_img_list) == 0
            raise ValueError('No frames found in the input video.')

        print('Detecting keypoints and smoothing alignment ...')
        avg_landmarks = None # Initialize
        if not has_aligned:
            raw_landmarks_list = [] # Changed name
            for i, img_np in enumerate(input_img_list): 
                self.face_helper.clean_all()
                self.face_helper.read_image(img_np) 
                num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, 
                    resize=640, 
                    eye_dist_threshold=5, 
                    only_keep_largest=True 
                )
                if num_det_faces == 1:
                    raw_landmarks_list.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
                elif num_det_faces >= 0: # Changed to include 0 detected faces
                    raw_landmarks_list.append(np.array([np.nan]*10))
            
            if not raw_landmarks_list: # No frames or no detections at all.
                 print("Warning: No raw landmarks were generated. Skipping landmark processing.")
                 avg_landmarks = np.full((len(input_img_list), 5, 2), np.nan) if input_img_list else np.empty((0,5,2))

            else:
                raw_landmarks_np = np.array(raw_landmarks_list) # Changed name
                if raw_landmarks_np.size > 0 : # Ensure not empty before processing
                    for i in range(raw_landmarks_np.shape[1]): # Iterate up to 10 (or actual number of columns)
                        raw_landmarks_np[:, i] = Predictor.interpolate_sequence(raw_landmarks_np[:, i])
                    video_length = len(input_img_list)
                    if raw_landmarks_np.ndim == 2 and raw_landmarks_np.shape[0] == video_length and raw_landmarks_np.shape[1] == 10:
                        # Filter only if sigma is less than half the length of the sequence to avoid errors
                        sigma_val = 5
                        if video_length <= sigma_val * 2 : # Heuristic to prevent filter error on short sequences
                            print(f"Warning: Video length ({video_length}) too short for Gaussian filter sigma ({sigma_val}). Using raw landmarks.")
                            avg_landmarks = raw_landmarks_np.reshape(video_length, 5, 2)
                        else:
                            avg_landmarks = gaussian_filter1d(raw_landmarks_np, sigma=sigma_val, axis=0).reshape(video_length, 5, 2)

                    elif video_length > 0 : 
                        print("Warning: Smoothed landmarks could not be computed as expected. Using interpolated raw landmarks or NaNs.")
                        # Attempt to reshape raw_landmarks_np if it's not empty, otherwise fill with NaNs
                        if raw_landmarks_np.size > 0 and raw_landmarks_np.shape[0] == video_length:
                             avg_landmarks = raw_landmarks_np.reshape(video_length, 5, 2) if raw_landmarks_np.shape[1] == 10 else np.full((video_length, 5, 2), np.nan)
                        else: # Fallback if shapes are totally off
                             avg_landmarks = np.full((video_length, 5, 2), np.nan)
                    else: 
                        avg_landmarks = np.empty((0,5,2))
                else: # raw_landmarks_np is empty
                    avg_landmarks = np.full((len(input_img_list), 5, 2), np.nan) if input_img_list else np.empty((0,5,2))


        cropped_faces_list = []
        for i, img_np in enumerate(input_img_list):
            self.face_helper.clean_all()
            self.face_helper.read_image(img_np)
            
            # Attempt to align and crop face
            perform_alignment = False
            if not has_aligned:
                if avg_landmarks is not None and i < len(avg_landmarks) and not np.isnan(avg_landmarks[i]).all():
                    self.face_helper.all_landmarks_5 = [avg_landmarks[i]]
                    perform_alignment = True
                else:
                    print(f"Warning: No valid landmarks for frame {i}. Attempting detection for alignment.")
                    # Try to detect landmarks directly for this frame if smoothed ones are bad
                    self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, resize=640, eye_dist_threshold=5, only_keep_largest=True)
                    if self.face_helper.all_landmarks_5:
                         perform_alignment = True
                    else:
                         print(f"Warning: No landmarks for frame {i} even after re-detection. Skipping face cropping for this frame.")

            elif has_aligned: # has_aligned = True
                if img_np.shape[0] != 512 or img_np.shape[1] != 512:
                     print(f"Warning: has_aligned is True, but image size is not 512x512. Resizing. Original shape: {img_np.shape}")
                     img_np_resized = cv2.resize(img_np, (512,512), interpolation=cv2.INTER_LINEAR)
                     self.face_helper.cropped_faces = [img_np_resized]
                else:
                     self.face_helper.cropped_faces = [img_np]
            
            if perform_alignment:
                 self.face_helper.align_warp_face()

            if not self.face_helper.cropped_faces: # If no face was cropped (either aligned or not)
                print(f"Warning: No cropped face for frame {i}. Using a black image placeholder.")
                dummy_face_t = torch.zeros(3, 512, 512, dtype=torch.float32) # Already on CPU
                normalize(dummy_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_faces_list.append(dummy_face_t)
                continue

            # Process the successfully cropped face
            cropped_face_np = self.face_helper.cropped_faces[0]
            cropped_face_t = img2tensor(cropped_face_np / 255., bgr2rgb=True, float32=True) # CPU tensor
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_faces_list.append(cropped_face_t)
        
        if not cropped_faces_list:
            raise ValueError("No faces could be cropped or processed from the video.")

        cropped_faces_batch = torch.stack(cropped_faces_list, dim=0).unsqueeze(0).to(self.device)
        
        print('Restoring faces ...')
        with torch.no_grad():
            video_length_actual = cropped_faces_batch.shape[1]
            output_frames_list = []
            for start_idx in range(0, video_length_actual, max_length):
                end_idx = min(start_idx + max_length, video_length_actual)
                
                # Prepare the input for the network. If the segment is a single frame, duplicate it.
                input_for_net = cropped_faces_batch[:, start_idx:end_idx, ...]

                if end_idx - start_idx == 1: # If the segment is a single frame
                    print("Segment is a single frame, duplicating for model input.")
                    # cropped_faces_batch is [1, total_frames, C, H, W]
                    # start_idx is the index of the single frame in this segment
                    input_for_net = cropped_faces_batch[:, [start_idx, start_idx], ...]
                
                processed_batch_segment = self.net(input_for_net, need_upscale=False)
                
                if end_idx - start_idx == 1: # If the original segment was 1 frame (and we fed 2 to model)
                    output_frames_list.append(processed_batch_segment[:, 0:1, ...]) # Take 1st frame from output
                else:
                    output_frames_list.append(processed_batch_segment)
            
            if not output_frames_list:
                    raise ValueError("Model did not produce any output frames.")

            output_combined_t = torch.cat(output_frames_list, dim=1).squeeze(0) # Still on device
            assert output_combined_t.shape[0] == video_length_actual, "Different number of frames in output"
            # Move to CPU for tensor2img and then to numpy
            restored_faces_np_list = [tensor2img(x.cpu(), rgb2bgr=True, min_max=(-1, 1)) for x in output_combined_t]
            del output_combined_t, cropped_faces_batch, output_frames_list, input_for_net, processed_batch_segment
            torch.cuda.empty_cache()

        print('Pasting faces back ...')
        restored_final_frames_list = []
        for i, original_img_np in enumerate(input_img_list): # Iterate over original CPU numpy images
            if i >= len(restored_faces_np_list):
                print(f"Warning: Not enough restored faces ({len(restored_faces_np_list)}) for input frames ({len(input_img_list)}). Stopping.")
                break

            self.face_helper.clean_all()
            current_restored_face_np = restored_faces_np_list[i].astype('uint8')

            final_frame_output = None
            if has_aligned:
                # For has_aligned, the output is the restored face, possibly upsampled.
                final_frame_output = current_restored_face_np
                if face_upsample and current_face_upsampler:
                    final_frame_output, _ = current_face_upsampler.enhance(final_frame_output, outscale=upscale)

            else: # Not has_aligned, requires pasting back to original frame context
                self.face_helper.read_image(original_img_np)
                
                if avg_landmarks is not None and i < len(avg_landmarks) and not np.isnan(avg_landmarks[i]).all():
                    original_cropped_faces = self.face_helper.cropped_faces.copy() if self.face_helper.cropped_faces else []

                    self.face_helper.all_landmarks_5 = [avg_landmarks[i]]
                    self._calculate_affine_matrices([avg_landmarks[i]])
                else:
                    print(f"Warning: No valid smoothed landmarks for frame {i} during paste-back. Attempting re-detection.")
                    original_cropped_faces = self.face_helper.cropped_faces.copy() if self.face_helper.cropped_faces else []

                    self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, resize=640, eye_dist_threshold=5, only_keep_largest=True)
                    if self.face_helper.all_landmarks_5:
                        # Calculate just the affine matrices without warping
                        self._calculate_affine_matrices(self.face_helper.all_landmarks_5)
                    else:
                        print(f"Warning: Cannot get landmarks for frame {i} for pasting. Pasting centrally.")
                        # Fallback: paste centrally if no affine transform possible
                        h, w, _ = original_img_np.shape
                        rh, rw, _ = current_restored_face_np.shape
                        x_c = (w - rw) // 2
                        y_c = (h - rh) // 2
                        final_frame_output = original_img_np.copy()
                        if x_c >=0 and y_c >=0 and x_c+rw <= w and y_c+rh <=h:
                             final_frame_output[y_c:y_c+rh, x_c:x_c+rw] = current_restored_face_np
                        else: # If restored face is larger than original, just use restored face
                             final_frame_output = current_restored_face_np
                        restored_final_frames_list.append(final_frame_output)
                        
                        self.face_helper.cropped_faces = original_cropped_faces
                        continue

                self.face_helper.add_restored_face(current_restored_face_np)
                
                # Restore the original state of cropped_faces to avoid conflicts
                self.face_helper.cropped_faces = original_cropped_faces
                
                bg_img_for_paste = None
                if current_bg_upsampler:
                    bg_img_for_paste, _ = current_bg_upsampler.enhance(original_img_np, outscale=upscale) 
                else: 
                    bg_img_for_paste = original_img_np.copy() # Use a copy

                self.face_helper.get_inverse_affine(None) 
                
                paste_face_upsampler_instance = current_face_upsampler if face_upsample else None
                
                final_frame_output = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img_for_paste, 
                    draw_box=draw_box, 
                    face_upsampler=paste_face_upsampler_instance
                )

            if final_frame_output is not None:
                restored_final_frames_list.append(final_frame_output)
            else:
                # Fallback: if something went wrong, append original or black.
                print(f"Warning: final_frame_output was None for frame {i}. Appending original frame.")
                restored_final_frames_list.append(original_img_np)


        if not restored_final_frames_list:
             raise ValueError("No frames were processed and restored after pasting.")

        print('Saving video ...')
        output_video_filename = f"{clip_name}_restored.mp4"
        output_video_path_str = os.path.join(output_dir, output_video_filename)
        
        height, width = restored_final_frames_list[0].shape[:2]
        
        vidwriter = VideoWriter(output_video_path_str, height, width, fps)
        for f_idx, frame_to_write in enumerate(restored_final_frames_list):
            if frame_to_write is None:
                print(f"Error: Frame {f_idx} is None before writing. Skipping.")
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                vidwriter.write_frame(black_frame)
                continue
            vidwriter.write_frame(frame_to_write)
        vidwriter.close()
        
        print(f'All results are saved in {output_video_path_str}.')
        return Path(output_video_path_str)
