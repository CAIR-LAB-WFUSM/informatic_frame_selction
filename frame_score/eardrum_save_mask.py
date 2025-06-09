#%%
import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import torch.nn as nn
from torchvision import models
import numpy as np
import importlib
import argparse
import os
from numpy.linalg import lstsq
from scipy.linalg import orth
import voc12.dataloader
import dataset.datasets
from misc import torchutils, imutils
import cv2
from gradCAM import GradCAM
import net.resnet50_cam
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
#%%


cudnn.enabled = True

class Args:
    # Environment
    num_workers = 1
    voc12_root = 'Dataset/VOC2012_SEG_AUG/'  # Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.

    # Dataset
    train_list = "voc12/train.txt"
    val_list = "voc12/val.txt"
    infer_list = "voc12/train.txt"  # voc12/train_aug.txt to train a fully supervised model,
                                    # voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.
    chainer_eval_set = "train"

    # Class Activation Map
    cam_network = "net.resnet50_cam"
    cam_crop_size = 512
    cam_batch_size = 1  # original: 16
    cam_num_epoches = 5
    cam_learning_rate = 0.1
    cam_weight_decay = 1e-4
    cam_eval_thres = 0.15
    cam_scales = (1.0, 0.5, 1.5, 2.0)  # Multi-scale inferences

    cam_weights_name = "sess/res50_cam.pth"
    target_layer = "stage4"
    cam_out_dir = "result/BC_AdvCAM" # positive: in datasets.py the label is [0,1], negative: in dataset.py the label is [1,0]
    adv_iter = 27
    AD_coeff = 7
    AD_stepsize = 0.08
    score_th = 0.5

# Now you can access the arguments like this:
args = Args()

# parser = argparse.ArgumentParser()

# # Environment
# parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
# parser.add_argument("--voc12_root", default='Dataset/VOC2012_SEG_AUG/', type=str,
#                     help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
# # Dataset
# parser.add_argument("--train_list", default="voc12/train.txt", type=str)
# parser.add_argument("--val_list", default="voc12/val.txt", type=str)
# parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
#                     help="voc12/train_aug.txt to train a fully supervised model, "
#                          "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
# parser.add_argument("--chainer_eval_set", default="train", type=str)

# # Class Activation Map
# parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
# parser.add_argument("--cam_crop_size", default=512, type=int)
# parser.add_argument("--cam_batch_size", default=2, type=int) # original: 16
# parser.add_argument("--cam_num_epoches", default=5, type=int)
# parser.add_argument("--cam_learning_rate", default=0.1, type=float)
# parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
# parser.add_argument("--cam_eval_thres", default=0.15, type=float)
# parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
#                     help="Multi-scale inferences")

# parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
# parser.add_argument("--target_layer", default="stage4")
# parser.add_argument("--cam_out_dir", default="result/cam_adv_mask", type=str)
# parser.add_argument("--adv_iter", default=27, type=int)
# parser.add_argument("--AD_coeff", default=7, type=int)
# parser.add_argument("--AD_stepsize", default=0.08, type=float)
# parser.add_argument("--score_th", default=0.5, type=float)

# args = parser.parse_args()
torch.set_num_threads(1)
if not os.path.exists(args.cam_out_dir):
    os.makedirs(args.cam_out_dir)

def process_and_compute_blurriness(binary_mask, original_image):
    # Ensure inputs are NumPy arrays
    if isinstance(original_image, torch.Tensor):
        # original_image = original_image.permute(1, 2, 0).numpy()  # Convert PyTorch tensor to NumPy array (H, W, C)
        original_image = original_image.numpy()
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.numpy()  # Convert PyTorch tensor to NumPy array

    original_image = original_image.transpose(1,2,0)
    # Squeeze the binary mask to remove extra dimensions
    if binary_mask.ndim == 3 and binary_mask.shape[-1] == 1:
        binary_mask = np.squeeze(binary_mask, axis=-1)  # Shape becomes (224, 224)
    
    # Mask the original image

    masked_image = original_image * binary_mask[:, :, None]  # Apply binary mask to each channel

    # Find the bounding box of the masked region
    coords = np.column_stack(np.where(binary_mask > 0))  # Get non-zero mask coordinates
    if coords.size == 0:
        raise ValueError("The binary mask is empty, no region to crop.")
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # Crop the bounding box
    cropped_image = masked_image[x_min:x_max + 1, y_min:y_max + 1, :]

    # Resize the cropped image back to 224x224
    resized_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Convert resized image to grayscale
    gray_image = cv2.cvtColor((resized_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Compute the blurriness of the resized image using the Laplacian variance
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    return resized_image, laplacian_var

def adv_climb(image, epsilon, data_grad):
    sign_data_grad = data_grad / (torch.max(torch.abs(data_grad))+1e-12)
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, image.min().data.cpu().float(), image.max().data.cpu().float()) # min, max from data normalization
    return perturbed_image

def add_discriminative(expanded_mask, regions, score_th):
    region_ = regions / regions.max()
    expanded_mask[region_>score_th]=1
    return expanded_mask

def get_largest_connected_component(binary_mask):
    """
    Retains only the largest connected component in the binary mask.
    """
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    
    # Find the largest component (excluding background, label 0)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = (labels == largest_label).astype(np.uint8)
        return largest_component
    return binary_mask  # If no components are found, return the original

def _work(process_id, model,  dataset_positive, dataset_negative, video_folder, args):
    try:
        databin_pos = dataset_positive[process_id]
        databin_neg = dataset_negative[process_id]
        n_gpus = torch.cuda.device_count()
        data_loader_pos = DataLoader(databin_pos, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=True)
        data_loader_neg = DataLoader(databin_neg, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=True)
        cam_sizes = [[], [], [], []] # scale 0,1,2,3
        with cuda.device(process_id):
            model.cuda()
            gcam = GradCAM(model=model, candidate_layers=[args.target_layer])
            highres_cam_pos = {}
            highres_cam_neg = {}
            org_img_dict = {}
            img_single, outputs, regions = None, None, None
            for idx_loader, data_loader in enumerate([data_loader_pos, data_loader_neg]):
                for iter, pack in enumerate(data_loader):
                    img_name = pack['name'][0]
                    name_appen = ['pos','neg']
                    if os.path.exists(os.path.join(video_folder, name_appen[idx_loader] + img_name + '.npy')):
                        print(f"{os.path.join(video_folder, name_appen[idx_loader] + img_name + '.npy')} exist")
                        continue
                    size = pack['size']
                    strided_size = imutils.get_strided_size(size, 4)
                    strided_up_size = imutils.get_strided_up_size(size, 16)
                    outputs_cam = []
                    n_classes = len(list(torch.nonzero(pack['label'][0])[:, 0]))

                    for s_count, size_idx in enumerate([1, 0, 2, 3]):
                        orig_img = pack['img'][size_idx].clone()
                        for c_idx, c in enumerate(list(torch.nonzero(pack['label'][0])[:, 0])):
                            pack['img'][size_idx] = orig_img
                            img_single = pack['img'][size_idx].detach()[0]  # [:, 1]: flip

                            if size_idx != 1:
                                total_adv_iter = args.adv_iter
                            else:
                                if args.adv_iter > 10:
                                    total_adv_iter = args.adv_iter // 2
                                    mul_for_scale = 2
                                elif args.adv_iter < 6:
                                    total_adv_iter = args.adv_iter
                                    mul_for_scale = 1
                                else:
                                    total_adv_iter = 5
                                    mul_for_scale = float(total_adv_iter) / 5

                            for it in range(total_adv_iter):
                                img_single.requires_grad = True

                                outputs = gcam.forward(img_single.cuda(non_blocking=True))

                                if c_idx == 0 and it == 0:
                                    cam_all_classes = torch.zeros([n_classes, outputs.shape[2], outputs.shape[3]])

                                gcam.backward(ids=c)

                                regions = gcam.generate(target_layer=args.target_layer)
                                regions = regions[0] + regions[1].flip(-1)

                                if it == 0:
                                    init_cam = regions.detach()

                                cam_all_classes[c_idx] += regions[0].data.cpu() * mul_for_scale
                                logit = outputs
                                logit = F.relu(logit)
                                logit = torchutils.gap2d(logit, keepdims=True)[:, :, 0, 0]

                                valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                                logit_loss = - 2 * (logit[:, c]).sum() + torch.sum(logit)

                                expanded_mask = torch.zeros(regions.shape)
                                expanded_mask = add_discriminative(expanded_mask, regions, score_th=args.score_th)

                                L_AD = torch.sum((torch.abs(regions - init_cam))*expanded_mask.cuda())

                                loss = - logit_loss - L_AD * args.AD_coeff

                                model.zero_grad()
                                img_single.grad.zero_()
                                loss.backward()

                                data_grad = img_single.grad.data

                                perturbed_data = adv_climb(img_single, args.AD_stepsize, data_grad)
                                img_single = perturbed_data.detach()

                        outputs_cam.append(cam_all_classes)

                    strided_cam = torch.sum(torch.stack(
                        [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                        in outputs_cam]), 0)
                    highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                                mode='bilinear', align_corners=False) for o in outputs_cam]

                    highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
                    strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
                    highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

                    org_img = pack['img'][0].detach()[0][0]
                    # Convert the PyTorch tensor to a PIL image
                    # Ensure the tensor is in the format (H, W, C) and scaled to [0, 255]
                    org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min())  # Normalize to [0, 1]
                    org_img = (org_img * 255).byte()  # Scale to [0, 255] and convert to uint8 for image compatibility

                    org_img_pil = transforms.ToPILImage()(org_img)

                    # Define the resizing transform
                    resize_transform = transforms.Resize((224, 224))

                    # Apply the resizing transform
                    resized_org_img = resize_transform(org_img_pil)

                    # Optionally, convert the resized image back to a PyTorch tensor
                    resized_org_img_tensor = transforms.ToTensor()(resized_org_img)


                    if idx_loader == 0:
                        pos_cam=highres_cam.cpu().numpy()
                        mask_path = os.path.join(video_folder, 'pos' + img_name + '.npy')
                        np.save(mask_path, pos_cam)
                        print(f"positive mask for image {img_name} has been saved in {mask_path}")
                    else:
                        neg_cam=highres_cam.cpu().numpy()
                        mask_path = os.path.join(video_folder, 'neg' + img_name + '.npy')
                        np.save(mask_path, neg_cam)
                        print(f"negative mask for image {img_name} has been saved in {mask_path}")
        if img_single is not None:
            del img_single
        if outputs is not None:
            del outputs
        if regions is not None:
            del regions
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"Error in process {process_id}: {e}")
        traceback.print_exc()
            

#%%



if __name__ == '__main__':
    model_path = './model_weights/best_resnet50_eardrum.pth'
    

    # Load the model and apply weights
    model = net.resnet50_cam.CAM(out_dim=2)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    n_gpus = torch.cuda.device_count()
    print(n_gpus)

    root_folder = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames'
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            if dir_name.endswith('.MOV'):
                video_folder = os.path.join(root, dir_name)

                dataset_positive = dataset.datasets.ImageDatasetMSFPositive( image_dir=video_folder, 
                                                            scales=args.cam_scales)
                dataset_negative = dataset.datasets.ImageDatasetMSFNegative( image_dir=video_folder, 
                                                            scales=args.cam_scales)
                
                # Set a fixed random seed to ensure consistent splitting
                random_seed = 42  # You can choose any fixed value
                torch.manual_seed(random_seed)

                # Split both datasets in the same way
                split_positive = torchutils.split_dataset(dataset_positive, n_gpus)
                torch.manual_seed(random_seed)  # Reset the seed before splitting the second dataset
                split_negative = torchutils.split_dataset(dataset_negative, n_gpus)

                # Use multiprocessing.spawn with the split datasets
                multiprocessing.spawn(
                    _work,
                    nprocs=n_gpus,
                    args=(model, split_positive, split_negative, video_folder, args),
                    join=True
                )
                
# %%
