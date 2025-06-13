# Otoscope Informative Frame Selector

This project aims to automatically select the most **informative frame** from an otoscope video. Informative frames are essential for downstream tasks such as diagnosis and training machine learning models. This script ranks video frames based on a combination of three criteria.

## Criteria for Informative Frame Selection

1. **Full Eardrum Visibility** – The frame should capture the complete eardrum region.
2. **Large Coverage** – The eardrum should occupy a large portion of the frame.
3. **Clarity** – The eardrum should be in sharp focus (low blurriness).

## Folder Structure

The input videos should be preprocessed into folders containing extracted frames in the following structure:

```plaintext
root_folder/ 
├── Effusion/ 
│ ├── ABC123.MOV/ 
│ │ ├── 1.png 
│ │ ├── 2.png 
│ │ └── ... 
│ └── ... 
├── Normal/ 
└── ...
```

Each `.MOV` folder should contain PNG images representing frames extracted from the original video.

## Usage (Batch Processing)

1. Clone the repository and prepare your dataset as described above.
2. Modify the `root_folder` variable in [`./frame_score/main.py`](./frame_score/main.py) to point to your dataset directory.
3. Run the main script:

```bash
python ./frame_score/main.py
After execution, each *.MOV/ folder will contain a file named ranked_scores.csv.
```

## Output: ranked_scores.csv
Each CSV contains the ranking information of frames with the following columns:

img	score	blurriness	Area	score_rank	blurriness_rank	Area_rank	weighted_rank
48.png	1.000000016	1309.83	16780	3	46	6	20.8
51.png	1.000000010	2404.21	12941	8	11	98	27.2

Higher weighted_rank indicates a more informative frame.

## Usage (Signle Video Processing)
You can also process a single .avi, .mov, or .mp4 video file without preparing extracted frames manually. Use the single_video_frame_evaluate.py script for this purpose.

### Example
```bash
python single_video_frame_evaluate.py
```

Make sure to modify the video_file and output_csv variables inside single_video_frame_evaluate.py:
```python
video_file = "/path/to/your/video/ABC123.MOV"
output_csv = "./results/ABC123_score.csv"
debug_mode = True  # Set to False to auto-delete temp folder
```
-This script will automatically extract frames, process them, and generate a CSV file with ranked scores.

-If debug_mode = True, intermediate files will be kept in a temp folder under the working directory for inspection.

## License
This project is licensed under the MIT License.
