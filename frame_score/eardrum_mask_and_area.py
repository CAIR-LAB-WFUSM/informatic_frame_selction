#%%
import os
import numpy as np
import pandas as pd
def main(root_folder):
    
    # Iterate through all subfolders in the base folder
    for subdir, _, _ in os.walk(root_folder):
        if not subdir.endswith(".MOV"):
            continue  # Skip directories that do not end with .MOV
        
        # Initialize a list to store mask areas for the current .MOV folder
        data = []
        
        # Load positive and negative npy files
        pos_files = sorted([f for f in os.listdir(subdir) if f.startswith("pos") and f.endswith(".npy")], key=lambda x: int(x.split('pos')[1].split('.')[0]))
        neg_files = sorted([f for f in os.listdir(subdir) if f.startswith("neg") and f.endswith(".npy")], key=lambda x: int(x.split('neg')[1].split('.')[0]))
        
        # Dictionary to store loaded numpy arrays
        pos_data = {f: np.load(os.path.join(subdir, f)) for f in pos_files}
        neg_data = {f: np.load(os.path.join(subdir, f)) for f in neg_files}
        
        # Compute the mask for each positive sample
        for pos_name, pos in pos_data.items():
            # Find corresponding negative samples
            neg_key = pos_name.replace("pos", "neg")  # Assuming neg and pos share the same index
            neg = neg_data.get(neg_key, np.zeros_like(pos))  # Default to zeros if no matching neg
            
            # Compute the hard threshold mask
            hard_threshold_mask = (1 * pos - 0.1 * neg - 0.6 > 0).astype(np.uint8)
            
            # Save the mask
            save_path = os.path.join(subdir, f"mask_{pos_name[3:]}")
            np.save(save_path, hard_threshold_mask)
            print(f"Saved mask: {save_path}")
            
            # Compute area of 1s in the mask
            area = np.sum(hard_threshold_mask)
            data.append([pos_name[3:], area])
        
        # Save results to a CSV file in the current .MOV folder
        csv_path = os.path.join(subdir, "area.csv")
        df = pd.DataFrame(data, columns=["Mask_Name", "Area"])
        df.to_csv(csv_path, index=False)
        print(f"Saved area data to: {csv_path}")

    print("Processing complete.")

if __name__ == '__main__':
    root_folder = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames'
    main(root_folder)
# %%
