import pandas as pd
from scipy.stats import rankdata
import os

def merge_csvs(scores_file, area_file, output_file):
    # Load scores.csv
    scores_df = pd.read_csv(scores_file)
    
    # Load area.csv
    area_df = pd.read_csv(area_file)
    
    # Rename Mask_Name column to match the format in scores.csv
    area_df['img'] = area_df['Mask_Name'].str.replace('.npy', '', regex=False)
    
    # Merge the two dataframes on the 'img' column
    merged_df = scores_df.merge(area_df[['img', 'Area']], on='img', how='inner')
    
    # Save the merged dataframe
    merged_df.to_csv(output_file, index=False)

def process_folders_merge(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name.endswith('.MOV'):
                video_folder = os.path.join(root, dir_name)
                scores_path = os.path.join(video_folder, 'scores.csv')
                area_path = os.path.join(video_folder, 'area.csv')
                output_path = os.path.join(video_folder, 'merged_scores_v1.csv')
                
                merge_csvs(scores_path, area_path, output_path)
                print(f"Merged file saved to: {output_path}")

# Define the function to process the ranking
def compute_weighted_rank(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Compute ranks (lower rank means higher value in original data)
    df["score_rank"] = rankdata(-df["score"], method="average")
    df["blurriness_rank"] = rankdata(-df["blurriness"], method="average")
    df["Area_rank"] = rankdata(-df["Area"], method="average")

    # Define weights for weighted average rank
    w_score, w_blurriness, w_area = 0.4, 0.4, 0.2

    # Compute weighted rank score
    df["weighted_rank"] = (
        w_score * df["score_rank"] +
        w_blurriness * df["blurriness_rank"] +
        w_area * df["Area_rank"]
    )

    # Rank by weighted rank score (ascending order since lower rank is better)
    df_ranked = df.sort_values(by="weighted_rank").reset_index(drop=True)

    # Save the ranked table to a new CSV file
    df_ranked.to_csv(output_file, index=False)
    return output_file

def process_folders_rank(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name.endswith('.MOV'):
                video_folder = os.path.join(root, dir_name)
                scores_path = os.path.join(video_folder, 'merged_scores_v1.csv')
                output_path = os.path.join(video_folder, 'ranked_scores.csv')
                
                compute_weighted_rank(scores_path, output_path)
                print(f"Merged file saved to: {output_path}")
                
def main(base_directory):
    process_folders_merge(base_directory)
    process_folders_rank(base_directory)
    
if __name__ == "__main__":
    # Define input and output file paths
    base_directory = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames'  # Change this to your actual base folder
    main(base_directory)
