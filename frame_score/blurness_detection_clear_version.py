#%%
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from skimage import exposure
import pywt
import numpy as np

EPS = 1e-8
NUM_WORKERS = min(56, cpu_count())
print(f"{NUM_WORKERS} is using")

def crop_to_patch_size(img, patch_size):
    h, w = img.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    cropped = img[top:top + new_h, left:left + new_w]
    return cropped

def extract_valid_patches_vectorized(gray, mask, patch_size=100, threshold_ratio=0.05):
    gray = crop_to_patch_size(gray, patch_size=patch_size)
    mask = crop_to_patch_size(mask, patch_size=patch_size)
    h, w = gray.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Image size must be divisible by patch_size"

    # Reshape into (n_patches_y, patch_size, n_patches_x, patch_size) then transpose to (n_patches_y, n_patches_x, patch_size, patch_size)
    def split_to_patches(arr):
        return arr.reshape(h // patch_size, patch_size, w // patch_size, patch_size).transpose(0, 2, 1, 3)

    gray_patches = split_to_patches(gray)
    mask_patches = split_to_patches(mask)

    # Compute mask ratio per patch
    mask_ratios = mask_patches.sum(axis=(2, 3)) / (patch_size * patch_size)

    # Create boolean mask to keep valid patches
    keep = mask_ratios >= (1 - threshold_ratio)

    valid_patches = []
    for i in range(keep.shape[0]):
        for j in range(keep.shape[1]):
            if keep[i, j]:
                x, y = j * patch_size, i * patch_size
                patch = gray_patches[i, j] * mask_patches[i, j]
                valid_patches.append(((x, y), patch.copy()))

    return valid_patches


#===  use gray scale image =====
def generate_mask(img_path):
    # === Load image ===
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # === Resize keeping aspect ratio ===
    scale = 224.0 / max(h, w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    gray_resized = cv2.resize(gray, (resized_w, resized_h))

    # === Enhance contrast and blur ===
    gray_eq = exposure.equalize_adapthist(gray_resized / 255.0) * 255
    gray_eq = gray_eq.astype(np.uint8)
    blurred = cv2.medianBlur(gray_eq, 5)

    # === Hough Circle Detection ===
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=70, param2=30,
                               minRadius=40, maxRadius=int(max(resized_h, resized_w) * 0.5))

    best_score = -np.inf
    best_circle = None
    Y, X = np.ogrid[:resized_h, :resized_w]

    if circles is not None:
        for c in np.uint16(np.around(circles[0])):
            cx, cy, r = c
            if abs(cx - resized_w//2) > 40 or abs(cy - resized_h//2) > 20:
                continue
            mask_in = (X - cx)**2 + (Y - cy)**2 <= r**2
            mask_out = ((X - cx)**2 + (Y - cy)**2 <= (r+12)**2) & (~mask_in)
            # mask_out = ~mask_in
            if mask_in.sum() < 100: #or mask_out.sum() < 100:
                continue
            mean_in = gray_resized[mask_in].mean()
            mean_out = gray_resized[mask_out].mean()
            score = mean_in - mean_out
            if score > best_score:
                best_score = score
                best_circle = (cx, cy, r)

    # === Generate circular mask ===
    mask_circle = np.zeros((resized_h, resized_w), dtype=bool)
    if best_circle is not None:
        cx, cy, r = best_circle
        mask_circle = (X - cx)**2 + (Y - cy)**2 <= r**2

    # === Resize mask back to original size ===
    mask_full = cv2.resize(mask_circle.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    # === Bright region removal on full-res ===
    gray_eq_fullres = exposure.equalize_adapthist(gray / 255.0) * 255
    gray_eq_fullres = gray_eq_fullres.astype(np.uint8)

    # Bright regions (e.g. reflections)
    _, bright_mask = cv2.threshold(gray_eq_fullres, 200, 255, cv2.THRESH_BINARY)
    bright_mask = bright_mask > 0

    # Dark regions (e.g. shadows, obstructions)
    _, dark_mask = cv2.threshold(gray_eq_fullres, 30, 255, cv2.THRESH_BINARY_INV)
    dark_mask = dark_mask > 0

    # Combine both
    remove_mask = bright_mask | dark_mask

    final_mask = mask_full & (~remove_mask)
    #膨胀操作：让 mask 向外扩张一圈，减小边缘效应带来的高频
    kernel = np.ones((19, 19), np.uint8)  # 结构元大小可调
    final_mask = cv2.erode(final_mask.astype(np.uint8), kernel, iterations=1)

    # 返回中心和半径（缩放回原图尺寸）
    if best_circle is not None:
        scale_inv = max(h, w) / 224.0
        cx_full = int(best_circle[0] * scale_inv)
        cy_full = int(best_circle[1] * scale_inv)
        r_full = int(best_circle[2] * scale_inv)
        best_circle_full = (cx_full, cy_full, r_full)
    else:
        best_circle_full = (w // 2, h // 2, int(min(h, w) * 0.45))

    return final_mask.astype(np.uint8), gray, best_circle_full

# 修改后的 process_image 函数
def process_image(args):
    subdir, row = args
    img_filename = row["img"]
    img_path = os.path.join(subdir, img_filename)
    mask_path = os.path.join(subdir, f"mask_{img_filename}.npy")  # Corresponding mask file
    if not os.path.exists(img_path):
        return (row.name, None)
    mask_eardrum = np.load(mask_path)  # Load mask (binary numpy array)
    # Ensure mask is always 2D
    if mask_eardrum.ndim == 3 and mask_eardrum.shape[0] == 1:
        mask_eardrum = np.squeeze(mask_eardrum, axis=0)  # Convert (1, H, W) -> (H, W)
    if mask_eardrum.ndim != 2:
        print(f"Skipping {mask_path}: Invalid mask shape {mask_eardrum.shape}")
        return (row.name, None)
    
    # try:
    mask, patch_gray, circle  = generate_mask(img_path)
    # Step 1: Resize eardrum mask to match mask size
    if mask_eardrum.shape != mask.shape:
        mask_eardrum_resized = cv2.resize(mask_eardrum.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_eardrum_resized = mask_eardrum

    # # Step 2: Binarize to ensure it's boolean
    mask_eardrum_resized = (mask_eardrum_resized > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    # # Step 3: Combine masks
    mask_combined = mask & mask_eardrum_resized

    
    if circle is None:
        return (row.name, 0.0)  # 无法检测到有效圆，视为最模糊
    
    blurriness = compute_blur_score(patch_gray, mask_combined, circle)
    return (row.name, blurriness)



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

def find_focus_offset(gray, circle):
    # Create a focus map using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_map = np.abs(laplacian)
    focus_map_norm = cv2.normalize(focus_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Find the area with highest focus measure
    kernel_size = 50
    mean_focus = cv2.blur(focus_map_norm, (kernel_size, kernel_size))
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mean_focus)
    cx, cy, r = circle
    x, y = maxLoc  # maxLoc is (x, y) = (column, row)

    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    offset = (r - distance) / r
    return offset

def compute_blur_score(patch_gray, mask,circle, tau=25.0):
    EPS = 1e-8
    # patch_gray = cv2.resize(patch_gray, (224, 224))
    # mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    # === 有效区域比例 ===
    coverage = np.sum(mask) / mask.size
    if coverage < 0.1:
        return 0.0  # 遮挡严重，直接标0

    # === 原始 GMM-based q2 ===
    q2 = compute_gradient_hist_span(patch_gray, mask, tau=tau)
    offset = find_focus_offset(patch_gray, circle)
    blur_score = q2 * offset
    return blur_score


# Main function to iterate over subdirectories and process images in parallel
def main(root_folder, selected_png_dir=None):
    # root_folder = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames"

    selected_mov_folders = set()
    
    if selected_png_dir is not None:
        print(f"Filtering based on .png files in: {selected_png_dir}")
        for root, _, files in os.walk(selected_png_dir):
            for f in files:
                if f.endswith(".png"):
                    mov_name = os.path.splitext(f)[0] + ".MOV"
                    selected_mov_folders.add(mov_name)

    # Iterate through all subfolders
    for subdir, _, _ in os.walk(root_folder):
        if not subdir.endswith(".MOV"):
            continue  # Skip non-MOV directories
        
        mov_folder = os.path.basename(subdir)
        if selected_png_dir is not None and mov_folder not in selected_mov_folders:
            continue  # Skip if not selected

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



def single_video_main(video_dir):
    assert os.path.isdir(video_dir), f"{video_dir} is not a valid folder"

    scores_path = os.path.join(video_dir, "scores.csv")
    if not os.path.exists(scores_path):
        print("scores.csv file is missing")
        return

    df = pd.read_csv(scores_path)

    # Initialize "blurriness" column if missing
    if "blurriness" not in df.columns:
        df["blurriness"] = np.nan

    # Prepare output dir
    video_name = os.path.basename(video_dir)
    output_dir = os.path.join("./result", video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments
    args_list = [(video_dir, row) for _, row in df.iterrows()]
    print(f"{video_dir} is preparing")

    # Run sequentially (or use multiprocessing if preferred)
    results = []
    for args in args_list:
        result = process_image(args)
        results.append(result)

        # Save masked frame to output folder
        img_filename = args[1]["img"]
        src_img_path = os.path.join(video_dir, img_filename)
        dst_img_path = os.path.join(output_dir, img_filename)

        if os.path.exists(src_img_path):
            img = cv2.imread(src_img_path)
            mask, gray, circle = generate_mask(src_img_path)

            # Apply mask to image
            masked_img = img.copy()
            masked_img[mask == 0] = 0

            # Draw circle
            if circle is not None:
                cx, cy, r = circle
                cv2.circle(masked_img, (cx, cy), r, (0, 255, 0), 2)  # Green circle
                cv2.circle(masked_img, (cx, cy), 8, (0, 0, 255), 2)  # Red center dot
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_map = np.abs(laplacian)
            focus_map_norm = cv2.normalize(focus_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Find the area with highest focus measure
            kernel_size = 50
            mean_focus = cv2.blur(focus_map_norm, (kernel_size, kernel_size))
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mean_focus)
            cv2.circle(masked_img, maxLoc, radius=20, color=(0, 255, 0), thickness=3)
            cv2.imwrite(dst_img_path, masked_img)

    # Update dataframe with results
    for index, blurriness in results:
        if blurriness is not None:
            df.at[index, "blurriness"] = blurriness

    # Save updated scores.csv
    df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)
    print(f"Processed {video_dir} into {output_dir}")

#%%
if __name__ == "__main__":
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Normal/CE11L.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Normal/CE201R.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Effusion/AM340L.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Effusion/AM331R.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Effusion/AM340L.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Effusion/AM345L.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Effusion/AM357R.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Effusion/OT082620.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Effusion/CE17R.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Perforation/CE61R.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Tympanosclerosis/CE96R.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Normal/NT26L.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Normal/OT083113.MOV"
    # video_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames/Normal/OT131608.MOV"
    # single_video_main(video_path)

    root_folder = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames"
    selected_dir = "/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/human_selected_new_all"
    main(root_folder, selected_png_dir=selected_dir)
    # main()
