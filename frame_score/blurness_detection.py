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
#=== only use red channel ===
# def generate_mask(img_path):
#     # === Load image ===
#     img = cv2.imread(img_path)
#     red_channel = img[:, :, 2]  # Use red channel
#     h, w = red_channel.shape

#     # === Resize keeping aspect ratio ===
#     scale = 224.0 / max(h, w)
#     resized_h, resized_w = int(h * scale), int(w * scale)
#     red_resized = cv2.resize(red_channel, (resized_w, resized_h))

#     # === Enhance contrast and blur ===
#     red_eq = exposure.equalize_adapthist(red_resized / 255.0) * 255
#     red_eq = red_eq.astype(np.uint8)
#     blurred = cv2.medianBlur(red_eq, 5)

#     # === Hough Circle Detection ===
#     circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
#                                param1=50, param2=30,
#                                minRadius=20, maxRadius=int(min(resized_h, resized_w) * 0.5))

#     best_score = -np.inf
#     best_circle = None
#     Y, X = np.ogrid[:resized_h, :resized_w]

#     if circles is not None:
#         for c in np.uint16(np.around(circles[0])):
#             cx, cy, r = c
#             if abs(cx - resized_w//2) > 40 or abs(cy - resized_h//2) > 20:
#                 continue
#             mask_in = (X - cx)**2 + (Y - cy)**2 <= r**2
#             mask_out = ((X - cx)**2 + (Y - cy)**2 <= (r+8)**2) & (~mask_in)
#             if mask_in.sum() < 100 or mask_out.sum() < 100:
#                 continue
#             mean_in = red_resized[mask_in].mean()
#             mean_out = red_resized[mask_out].mean()
#             score = mean_in - mean_out
#             if score > best_score:
#                 best_score = score
#                 best_circle = (cx, cy, r)

#     # === Generate circular mask ===
#     mask_circle = np.zeros((resized_h, resized_w), dtype=bool)
#     if best_circle is not None:
#         cx, cy, r = best_circle
#         mask_circle = (X - cx)**2 + (Y - cy)**2 <= r**2

#     # === Resize mask back to original size ===
#     mask_full = cv2.resize(mask_circle.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

#     # === Bright region removal on full-res ===
#     red_eq_fullres = exposure.equalize_adapthist(red_channel / 255.0) * 255
#     red_eq_fullres = red_eq_fullres.astype(np.uint8)
#     _, bright_mask = cv2.threshold(red_eq_fullres, 200, 255, cv2.THRESH_BINARY)
#     bright_mask = bright_mask > 0

#     final_mask = mask_full & (~bright_mask)
#     # 膨胀操作：让 mask 向外扩张一圈，减小边缘效应带来的高频
#     kernel = np.ones((19, 19), np.uint8)  # 结构元大小可调
#     final_mask = cv2.erode(final_mask.astype(np.uint8), kernel, iterations=1)

#     return final_mask.astype(np.uint8), red_channel

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
                               minRadius=40, maxRadius=int(min(resized_h, resized_w) * 0.5))

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
    kernel = np.ones((9, 9), np.uint8)  # 结构元大小可调
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
    # mask_path = os.path.join(subdir, f"mask_{img_filename}.npy")  # Corresponding mask file
    if not os.path.exists(img_path):
        return (row.name, None)
    # mask_eardrum = np.load(mask_path)  # Load mask (binary numpy array)
    # Ensure mask is always 2D
    # if mask_eardrum.ndim == 3 and mask_eardrum.shape[0] == 1:
    #     mask_eardrum = np.squeeze(mask_eardrum, axis=0)  # Convert (1, H, W) -> (H, W)
    # if mask_eardrum.ndim != 2:
    #     print(f"Skipping {mask_path}: Invalid mask shape {mask_eardrum.shape}")
    #     return (row.name, None)
    
    # try:
    mask, patch_gray, circle  = generate_mask(img_path)
    # Step 1: Resize eardrum mask to match mask size
    # if mask_eardrum.shape != mask.shape:
    #     mask_eardrum_resized = cv2.resize(mask_eardrum.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    # else:
    #     mask_eardrum_resized = mask_eardrum

    # # Step 2: Binarize to ensure it's boolean
    # mask_eardrum_resized = (mask_eardrum_resized > 0).astype(np.uint8)
    # mask = (mask > 0).astype(np.uint8)

    # # Step 3: Combine masks
    # mask_combined = mask & mask_eardrum_resized
    # patch_gray = cv2.resize(patch_gray, (224, 224))
    # mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    if circle is None:
        return (row.name, 0.0)  # 无法检测到有效圆，视为最模糊
    
    blurriness = compute_blur_score_concentric(patch_gray, mask, circle)
    return (row.name, blurriness)
    # except Exception as e:
    #     print(f"Error processing {img_path}: {e}")
    #     return (row.name, None)


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

# def compute_blur_laplacian(gray, mask):
#     gray = cv2.resize(gray, (224, 224))
#     mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
#     lap = cv2.Laplacian(gray, cv2.CV_64F)
#     lap = lap[mask > 0]
#     return lap.var()

# def compute_tenengrad_blur(gray, mask):
#     gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     grad_mag_sq = gx**2 + gy**2
#     return grad_mag_sq[mask > 0].mean()  # 越大越清晰

# def compute_fft_blur(gray, mask):
#     f = np.fft.fft2(gray)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = np.abs(fshift)
#     hfreq = magnitude_spectrum[mask > 0]
#     return np.mean(hfreq[hfreq > np.percentile(hfreq, 80)])  # 越大越清晰

def compute_fft_lowpass_energy(gray, mask):
    EPS = 1e-8
    gray = gray.astype(np.float32)
    mask = mask.astype(bool)

    # 仅保留有效区域
    gray_masked = gray * mask

    # 进行 DFT（二维傅里叶），使用 window 减少泄漏
    window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
    gray_windowed = gray_masked * window

    f = np.fft.fft2(gray_windowed)
    fshift = np.fft.fftshift(f)
    power_spectrum = np.abs(fshift) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) // 8
    lowpass_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

    # 低频能量占比越大，说明图像越模糊
    total_energy = np.sum(power_spectrum)
    lowpass_energy = np.sum(power_spectrum[lowpass_mask])

    if total_energy < EPS:
        return 0.0

    ratio = lowpass_energy / total_energy
    return 1/(ratio+EPS)  # 越大越模糊

def compute_wavelet_energy(gray, mask, wavelet='haar', level=1):
    gray = gray.astype(np.float32)
    mask = mask.astype(bool)

    # 小波变换（2D）仅第 1 层，得到近似+细节系数
    coeffs2 = pywt.wavedec2(gray, wavelet=wavelet, level=level)
    cH, cV, cD = coeffs2[1]  # 水平、垂直、对角细节

    # Resize mask 匹配变换后的尺寸
    h_c, w_c = cH.shape
    h, w = gray.shape
    mask_resized = cv2.resize(mask.astype(np.uint8), (w_c, h_c), interpolation=cv2.INTER_NEAREST).astype(bool)

    # 在有效区域统计能量
    energy = (cH**2 + cV**2 + cD**2)[mask_resized].mean()
    # energy = (cD**2)[mask_resized].mean()
    # print((cV**2)[mask_resized])
    # print((cH**2)[mask_resized])
    # print((cD**2)[mask_resized])
    return energy  # 越大越清晰

def wavelet_blur_detect(gray, threshold=3):
    
    # Convert image to grayscale
    # Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Y = gray #* mask
    
    
    M, N = Y.shape
    
    # Crop input image to be 3 divisible by 2
    M_cropped, N_cropped = int(M / 16) * 16, int(N / 16) * 16
    Y = Y[:M_cropped, :N_cropped]
    # mask = mask[:M_cropped, :N_cropped]

    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    # # Resize mask for each level
    # mask1 = cv2.resize(mask.astype(np.uint8), E1.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    # mask2 = cv2.resize(mask.astype(np.uint8), E2.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    # mask3 = cv2.resize(mask.astype(np.uint8), E3.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

    # # 掩膜作用在边缘图上，保留二维结构
    # E1 = E1 * mask1
    # E2 = E2 * mask2
    # E3 = E3 * mask3
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold
    EdgePoint2 = Emax2 > threshold
    EdgePoint3 = Emax3 > threshold
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
            
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
    
        if EdgePoint[i] == 1:
        
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
            
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges))

    for i in range(n_edges):
    
        if RGstructure[i] == 1 or RSstructure[i] == 1:
        
            if Emax1[i] < threshold:
            
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, BlurExtent

# def compute_blur_score(patch_gray, mask, tau=25.0):
#     EPS = 1e-8
#     patch_gray = cv2.resize(patch_gray, (224, 224))
#     mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

#     # === 有效区域比例 ===
#     coverage = np.sum(mask) / mask.size
#     if coverage < 0.1:
#         return 0.0  # 遮挡严重，直接标0

#     # === 原始 GMM-based q2 ===
#     q2 = compute_gradient_hist_span(patch_gray, mask, tau=tau)

#     # === Laplacian variance ===
#     lap_var = compute_blur_laplacian(patch_gray, mask)

#     # === Tenengrad ===
#     Tenengrad = compute_tenengrad_blur(patch_gray, mask)

#     # === High frequence energy ===
#     fft_score = compute_fft_lowpass_energy(patch_gray, mask)

#     # === Wavelet energy ===
#     wavelet_energy = compute_wavelet_energy(patch_gray, mask)
#     # === 融合得分（越大越模糊） ===
#     a1 = 0
#     a2 = 0
#     a3 = 0
#     a4 = 1-a1-a2-a3
#     # blur_score = a1 * q2 +  a2 * lap_var + a3 * Tenengrad + a4 * fft_score 
#     blur_score = fft_score
#     return blur_score

def compute_blur_score_concentric(patch_gray, mask, circle, tau=25.0):
    EPS = 1e-8
    # patch_gray = cv2.resize(patch_gray, (224, 224))
    # mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    # === 有效区域比例 ===
    coverage = np.sum(mask) / mask.size
    if coverage < 0.1 or circle is None:
        return 0.0  # 遮挡严重或无圆信息，直接标0

    # === 获取圆心与半径 ===
    cx, cy, r = circle
    # scale = 224.0 / max(patch_gray.shape)
    # cx = int(cx * scale)
    # cy = int(cy * scale)
    # r = int(r * scale)

    
    # === 生成同心圆区域 ===
    h, w = patch_gray.shape
    Y, X = np.ogrid[:h, :w]
    blur_scores = []
    weights = [20,10,5,1,1,1,1]

    for i in range(7):
        inner_r = int(r * i / 5)
        outer_r = int(r * (i + 1) / 5)
        ring_mask = ((X - cx)**2 + (Y - cy)**2 <= outer_r**2) & ((X - cx)**2 + (Y - cy)**2 > inner_r**2)
        ring_mask = ring_mask & (mask > 0)

        if ring_mask.sum() <= 10:
            blur_scores.append(0)
        else:
            # 计算q2
            q2 = compute_gradient_hist_span(patch_gray, ring_mask)
            blur_scores.append(q2)

        # 权重，中心权重更高
        # weights.append(10 - 2*i)

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()
    blur_scores = np.array(blur_scores, dtype=np.float32)

    if len(blur_scores) != len(weights):
        raise Exception("weight, score lenght are not consist")  # 安全防御，避免 shape mismatch

    blur_scores = float(np.sum(blur_scores * weights))

    # valid_patches = extract_valid_patches_vectorized(patch_gray, mask, patch_size=200, threshold_ratio=0.001)
    # blur_scores = 0
    # len_patches = 0
    # for (x,y), patch in valid_patches:
    #     # per,_ = wavelet_blur_detect(patch) #, mask= np.ones(patch.shape))
    #     per = compute_gradient_hist_span(patch, mask= np.ones(patch.shape))
    #     weight = r**2 / ((x-cx)**2+(y-cy)**2+EPS)
    #     blur_scores += per * weight
    #     len_patches += 1

    # if len_patches != 0:
    #     blur_scores = blur_scores/len_patches
    # else:
    #     blur_scores = 0
    return  blur_scores

    # blur_scores = np.array(blur_scores, dtype=np.float32)
    # return float((blur_scores * weights).sum())

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

import shutil

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

            # # Load eardrum mask
            # mask_path = os.path.join(video_dir, f"mask_{img_filename}.npy")
            # if os.path.exists(mask_path):
            #     mask_eardrum = np.load(mask_path)
            #     if mask_eardrum.ndim == 3 and mask_eardrum.shape[0] == 1:
            #         mask_eardrum = np.squeeze(mask_eardrum, axis=0)
            #     if mask_eardrum.shape != mask.shape:
            #         mask_eardrum = cv2.resize(mask_eardrum.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            #     mask_eardrum = (mask_eardrum > 0).astype(np.uint8)
            # else:
            #     mask_eardrum = np.ones_like(mask, dtype=np.uint8)  # fallback to all-ones

            # mask = (mask > 0).astype(np.uint8)
            # mask_combined = mask & mask_eardrum

            # # Save debug mask image
            # debug_mask = (mask_combined * 255).astype(np.uint8)
            # debug_mask_path = os.path.join(output_dir, f"debug_combined_mask_{img_filename}")
            # cv2.imwrite(debug_mask_path, debug_mask)

            # Apply mask to image
            masked_img = img.copy()
            masked_img[mask == 0] = 0

            # Draw circle
            if circle is not None:
                cx, cy, r = circle
                cv2.circle(masked_img, (cx, cy), r, (0, 255, 0), 2)  # Green circle
                cv2.circle(masked_img, (cx, cy), 3, (0, 0, 255), -1)  # Red center dot

            cv2.imwrite(dst_img_path, masked_img)

            # # Save valid patches
            # frame_name = os.path.splitext(img_filename)[0]
            # patch_output_dir = os.path.join(output_dir, frame_name)
            # os.makedirs(patch_output_dir, exist_ok=True)

            # valid_patches = extract_valid_patches_vectorized(gray, mask, patch_size=200, threshold_ratio=0.1)
            # for idx, ((x, y), patch) in enumerate(valid_patches):
            #     patch_path = os.path.join(patch_output_dir, f"patch{idx+1}.png")
            #     cv2.imwrite(patch_path, patch)

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
    # single_video_main(video_path)
    main()
