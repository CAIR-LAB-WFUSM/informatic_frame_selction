from blurness_detection_clear_version import main as main_blurness
from eardrum_mask_and_area import main as main_area
from eardrum_save_mask import main as main_mask
from eardrum_score import main as main_eardrum_score
from rank_score import main as main_rank_score
root_folder = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_video_frames'

main_eardrum_score(root_folder)
main_mask(root_folder)
main_area(root_folder)
main_blurness(root_folder)
main_rank_score(root_folder)