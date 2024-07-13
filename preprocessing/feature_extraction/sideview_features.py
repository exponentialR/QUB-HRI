import sys
import os
from tqdm import tqdm
import logging
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from reconstruction.sideview_extraction import LandmarksToHDF5
from preprocessing.landmark_extraction import setup_calibration_video_logger

logger = setup_calibration_video_logger(
    "Subtask-Sideview-Features-Logger",
    format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %('
               'message)s',
    extra_attrs=['task_name', 'detail'],
    error_log_file='landmark_extraction_log.txt',
    levels_to_save={logging.DEBUG, logging.INFO},
    console_level=logging.INFO
)

CAMERA_PAIRS = {
    'CAM_LL_CAM_LR': ('CAM_LL', 'CAM_LR'),
    'CAM_UL_CAM_UR': ('CAM_UL', 'CAM_UR'),
    'CAM_UR_CAM_UL': ('CAM_UR', 'CAM_UL'),
    'CAM_UL_CAM_LL': ('CAM_UL', 'CAM_LL'),
    'CAM_AV_CAM_LL': ('CAM_AV', 'CAM_LL'),
    'CAM_AV_CAM_LR': ('CAM_AV', 'CAM_LR'),
    'CAM_AV_CAM_UL': ('CAM_AV', 'CAM_UL'),
    'CAM_AV_CAM_UR': ('CAM_AV', 'CAM_UR')
}


class SubtaskSideviewFeatures:
    def __init__(self, proj_dir, output_dir, camera_pairs='CAM_LL_CAM_LR'):
        self.proj_dir = proj_dir
        self.output_dir = output_dir
        self.camera_pairs = CAMERA_PAIRS[camera_pairs]

    def extract_landmarks(self):
        subtask_list = [folder for folder in os.listdir(self.proj_dir) if
                        os.path.isdir(os.path.join(self.proj_dir, folder))]
        total_count_landmark = 0
        for subtask in subtask_list:
            subtask_dir = os.path.join(self.proj_dir, subtask)
            current_subtask_files = [file for file in os.listdir(subtask_dir) if
                                     file.endswith('.mp4') and file.split('-')[1].startswith(self.camera_pairs)]
            os.makedirs(os.path.join(self.output_dir, subtask), exist_ok=True)
            subtask_count_landmark = 0
            for video_file in tqdm(current_subtask_files, desc='Processing subtask files'):
                video_path = os.path.join(subtask_dir, video_file)
                video_file = video_file.split('mp4')
                output_path = os.path.join(self.output_dir, subtask, video_file[0] + 'h5')
                print(output_path)
                # landmarks = LandmarksToHDF5(video_path, output_path, logger=logger)
                # landmarks.process_and_save()
                subtask_count_landmark += 1
            print(f'{subtask_count_landmark} files processed for subtask {subtask}')

            total_count_landmark += subtask_count_landmark
            logger.info(f'Landmarks extracted for subtask {subtask}',
                        extra={'task_name': 'Landmark Extraction', 'detail': 'Subtask Processing'})
        logger.info(f'Total landmarks extracted for all subtasks: {total_count_landmark}',
                    extra={'task_name': 'Landmark Extraction', 'detail': 'Total Processing'})


if __name__ == "__main__":
    proj_dir = '/home/samueladebayo/Documents/PhD/QUB-PHEO-Dataset/test-annotate'
    output_dir = '/home/samueladebayo/Documents/PhD/QUB-PHEO-Dataset/landmark'
    subtask_sideview_features = SubtaskSideviewFeatures(proj_dir, output_dir)
    subtask_sideview_features.extract_landmarks()
