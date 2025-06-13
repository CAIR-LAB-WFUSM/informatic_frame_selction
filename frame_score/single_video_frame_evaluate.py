import os
import shutil
import cv2
import pandas as pd
import tempfile
from datetime import datetime
import imageio
import time  # ⏱️ Added for timing

# Import your main functions
from blurness_detection_clear_version import main as main_blurness
from eardrum_mask_and_area import main as main_area
from eardrum_save_mask import main as main_mask
from eardrum_score import main as main_eardrum_score
from rank_score import main as main_rank_score

def validate_video(video_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
        cap.release()
        raise ValueError(f"Video file is not readable or contains no frames: {video_path}")
    cap.release()

def extract_frames(video_path, output_folder):
    reader = imageio.get_reader(video_path)
    for frame_id, frame in enumerate(reader, start=1):
        frame_path = os.path.join(output_folder, f"{frame_id}.png")
        imageio.imwrite(frame_path, frame)
    reader.close()

def process_video(video_path, output_csv_path, debug=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Validate input video
    validate_video(video_path)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Choose temp folder location
    temp_root = os.path.join(os.getcwd(), "temp_debug") if debug else tempfile.gettempdir()
    temp_dir_name = f"tmp_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dir = os.path.join(temp_root, temp_dir_name)
    os.makedirs(temp_dir, exist_ok=True)

    # ⏱️ Total time start
    total_start_time = time.time()
    try:
        # ⏱️ Step 1: Frame extraction
        t0 = time.time()
        # Extract frames into temp dir
        extract_frames(video_path, temp_dir)
        print(f"[⏱️] Frame extraction took {time.time() - t0:.2f} seconds")

        # Wrap frames into fake MOV-style subfolder
        fake_mov_dir = os.path.join(temp_dir, f"{video_name}.MOV")
        os.makedirs(fake_mov_dir, exist_ok=True)

        for fname in os.listdir(temp_dir):
            if fname.endswith(".png"):
                shutil.move(os.path.join(temp_dir, fname), os.path.join(fake_mov_dir, fname))

        # Run your pipeline
        # ⏱️ Step 2: Eardrum score
        t0 = time.time()
        main_eardrum_score(temp_dir)
        print(f"[⏱️] main_eardrum_score() took {time.time() - t0:.2f} seconds")

        # ⏱️ Step 3: Save mask
        t0 = time.time()
        main_mask(temp_dir)
        print(f"[⏱️] main_mask() took {time.time() - t0:.2f} seconds")

        # ⏱️ Step 4: Compute area
        t0 = time.time()
        main_area(temp_dir)
        print(f"[⏱️] main_area() took {time.time() - t0:.2f} seconds")

        # ⏱️ Step 5: Compute blurriness
        t0 = time.time()
        main_blurness(temp_dir)
        print(f"[⏱️] main_blurness() took {time.time() - t0:.2f} seconds")

        # ⏱️ Step 6: Rank score
        t0 = time.time()
        main_rank_score(temp_dir)
        print(f"[⏱️] main_rank_score() took {time.time() - t0:.2f} seconds")
        
        # Copy ranked_scores.csv and rename
        result_path = os.path.join(fake_mov_dir, "ranked_scores.csv")
        if os.path.exists(result_path):
            shutil.copy(result_path, output_csv_path)
        else:
            raise FileNotFoundError("ranked_scores.csv not found")

    finally:
        total_time = time.time() - total_start_time  # ⏱️ Total time end
        print(f"[⏱️] Total processing time: {total_time:.2f} seconds")
        # Clean up temp directory if not in debug mode
        if not debug:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"[DEBUG MODE] Temp folder preserved at: {temp_dir}")

if __name__ == "__main__":
    # Example usage
    video_file = "/isilon/datalake/gurcan_rsch/original/otoscope/Otoscope/otoscope_videos/2024_2025/NCH Images 02-03.2025/Elmaraghy Feb 2025/CE1115L.avi"
    output_csv = "./results/CE1115L.csv"
    debug_mode = True  # Set to False for run mode

    process_video(video_file, output_csv, debug=debug_mode)
