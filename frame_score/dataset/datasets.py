import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDatasetMSFNegative(Dataset):
    def __init__(self, image_dir, scales=(1.0,), transform=None):
        """
        Args:
            image_dir (str): Directory containing the images.
            scales (tuple): Scales for multi-scale processing.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.image_dir = image_dir
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.scales = scales
        self.transform = transform
        self.label = torch.tensor([1, 0])  # One-hot encoded label [0, 1].
            # Define the transformations
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB mode.
        
        image = image.resize((224,224))

        # Multi-scale image list
        ms_img_list = []
        for scale in self.scales:
            if scale == 1.0:
                scaled_img = image
            else:
                scaled_size = (int(image.width * scale), int(image.height * scale))
                scaled_img = image.resize(scaled_size, Image.BILINEAR)

            scaled_img = self.to_tensor(scaled_img)
            scaled_img = self.normalize(scaled_img)
            # Add both normal and flipped versions
            ms_img_list.append(torch.stack([scaled_img, torch.flip(scaled_img, [-1])], dim=0))

        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]  # Simplify if only one scale is used.

        output = {
            "name": os.path.basename(img_path),  # Image file name
            "img": ms_img_list,  # Multi-scale image tensor
            "size": (image.width,image.height),  # Original image size (width, height)
            "label": self.label  # Fixed one-hot encoded label [0, 1]
        }
        return output
    

class ImageDatasetMSFPositive(Dataset):
    def __init__(self, image_dir, scales=(1.0,), transform=None):
        """
        Args:
            image_dir (str): Directory containing the images.
            scales (tuple): Scales for multi-scale processing.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.image_dir = image_dir
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.scales = scales
        self.transform = transform
        self.label = torch.tensor([0, 1])  # One-hot encoded label [0, 1].
            # Define the transformations
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB mode.
        
        image = image.resize((224,224))

        # Multi-scale image list
        ms_img_list = []
        for scale in self.scales:
            if scale == 1.0:
                scaled_img = image
            else:
                scaled_size = (int(image.width * scale), int(image.height * scale))
                scaled_img = image.resize(scaled_size, Image.BILINEAR)

            scaled_img = self.to_tensor(scaled_img)
            scaled_img = self.normalize(scaled_img)
            # Add both normal and flipped versions
            ms_img_list.append(torch.stack([scaled_img, torch.flip(scaled_img, [-1])], dim=0))

        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]  # Simplify if only one scale is used.

        output = {
            "name": os.path.basename(img_path),  # Image file name
            "img": ms_img_list,  # Multi-scale image tensor
            "size": (image.width,image.height),  # Original image size (width, height)
            "label": self.label  # Fixed one-hot encoded label [0, 1]
        }
        return output