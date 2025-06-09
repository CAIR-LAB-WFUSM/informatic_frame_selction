import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import pandas as pd
from multiprocessing import Pool, cpu_count

EPS = 1e-8
NUM_WORKERS = min(56, cpu_count())  # Use all available CPUs but limit to 56
print(f"{NUM_WORKERS} is using")

# Define the function to process a single image
def process_image(args):
    subdir, row = args  # Unpack arguments
    img_filename = row["img"]  # Image filename (e.g., "1.png")
    img_path = os.path.join(subdir, img_filename)
    mask_path = os.path.join(subdir, f"mask_{img_filename}.npy")  # Corresponding mask file

    # Check if both the image and mask exist
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return (row.name, None)  # Return index with None if files are missing

    # Load image and mask
    patch_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    patch_gray = cv2.resize(patch_gray, (224, 224))
    mask = np.load(mask_path)  # Load mask (binary numpy array)

    # Ensure mask is always 2D
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)  # Convert (1, H, W) -> (H, W)
    if mask.ndim != 2:
        print(f"Skipping {mask_path}: Invalid mask shape {mask.shape}")
        return (row.name, None)

    # Compute blurriness
    blurriness = compute_gradient_hist_span(patch_gray, mask)
    
    return (row.name, blurriness)

# Function to compute gradient hist span
def compute_gradient_hist_span(patch_gray, mask, tau=25.0):
    gx = cv2.Sobel(patch_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    valid_pixels = mask == 1
    grad_mag_masked = grad_mag[valid_pixels].reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(grad_mag_masked)

    variances = [gmm.covariances_[i][0, 0] for i in range(2)]
    sigma1 = max(variances)

    patch_gray = np.clip(patch_gray, 0, 255).astype(np.float32)
    patch_gray = np.nan_to_num(patch_gray, nan=0.0, posinf=255.0, neginf=0.0)

    patch_gray_masked = patch_gray[valid_pixels]
    Lmax = np.max(patch_gray_masked)
    Lmin = np.min(patch_gray_masked)

    C_p = (Lmax - Lmin) / (Lmax + Lmin + EPS)
    q2 = (tau * sigma1) / (C_p + EPS)
    
    return q2

# Main function to iterate over subdirectories and process images in parallel
def main():
    root_folder = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames"
    
    # Iterate through all subfolders
    for subdir, _, _ in os.walk(root_folder):
        if not subdir.endswith(".MOV"):
            continue  # Skip non-MOV directories

        scores_path = os.path.join(subdir, "scores.csv")
        if not os.path.exists(scores_path):
            print("scores.csv file is missing")
            continue

        df = pd.read_csv(scores_path)

        # Initialize "blurriness" column if missing
        if "blurriness" not in df.columns:
            df["blurriness"] = np.nan

        # Prepare arguments for parallel processing
        args_list = [(subdir, row) for _, row in df.iterrows()]
        print(f"{subdir} is preparing")
        # Run parallel processing
        with Pool(NUM_WORKERS) as pool:
            results = pool.map(process_image, args_list)

        # Update dataframe with results
        for index, blurriness in results:
            if blurriness is not None:
                df.at[index, "blurriness"] = blurriness

        # Save updated scores.csv
        df.to_csv(scores_path, index=False)
        print(f"Processed {subdir}")

    print("Blurriness computation completed for all folders.")

if __name__ == "__main__":
    main()
