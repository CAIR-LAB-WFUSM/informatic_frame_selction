#%%
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision import models
import net.resnet50_cam



def is_valid_image_file(filename):
    # Check if the file has a valid numeric prefix and supported image extension
    parts = filename.split('.')
    return parts[0].isdigit() and filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def numeric_prefix_sort_key(filename):
    # Extract the numeric prefix safely
    return int(filename.split('.')[0])

def compute_scores(image_folder, model, transform, device):
    scores = []

    files = os.listdir(image_folder)
    # Filter valid image files
    valid_files = [f for f in files if is_valid_image_file(f)]

    # Sort based on numeric prefix
    sorted_files = sorted(valid_files, key=numeric_prefix_sort_key)
    for img_name in sorted_files:
        if img_name.endswith('.png'):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files
            img_path = os.path.join(image_folder, img_name)
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                eardrum_score = probabilities[0, 1].item()  # Probability for class 1 (eardrum)

            scores.append((img_name, eardrum_score))
            print(f'image_name:{img_name}, score:{eardrum_score}')
    return scores

# Process each video folder
def process_all_videos(root_folder, model, transform, device):
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            if dir_name.endswith('.MOV'):
                video_folder = os.path.join(root, dir_name)

                scores = compute_scores(video_folder, model, transform, device)

                # Save the scores to a CSV file
                output_file = os.path.join(video_folder, 'scores.csv')
                df = pd.DataFrame(scores, columns=['img', 'score'])
                df.to_csv(output_file, index=False)
                print(f'finished {dir_name}')

def main():
    # Define the transform for preprocessing the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),         # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
    ])

    # Load the pre-trained ResNet50 model and update the final layer
    resnet50 = net.resnet50_cam.Net(out_dim=2)
    # Load the trained weights
    model_path = 'best_resnet50_eardrum.pth'
    resnet50.load_state_dict(torch.load(model_path))
    resnet50.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50.to(device)  # Set the model to evaluation mode
    # Define the function to compute eardrum scores
    # Specify the root folder containing all video frames
    root_folder = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames'
    process_all_videos(root_folder, resnet50, transform, device)

    print("Eardrum scores computed and saved for all videos.")

# %%
if __name__ == '__main__':
    main()