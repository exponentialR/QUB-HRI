from preprocess_landmark import LandmarksToHDF5
import os
from tqdm import tqdm
from logger_utils import setup_calibration_video_logger
import logging

logger = setup_calibration_video_logger(
    "Landmark-Extraction-Logger",
    format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %('
               'message)s',
    extra_attrs=['task_name', 'detail'],
    error_log_file='landmark_extraction_log.txt',
    levels_to_save={logging.DEBUG, logging.INFO},
    console_level=logging.INFO
)


class SubtaskLandmarkExtractor:
    def __init__(self, proj_directory, start_part, end_part, logger=None, track_confidence=0.5,
                 detect_confidence=0.5):
        self.start_participant = start_part
        self.end_participant = end_part
        self.logger = logger
        self.proj_dir = proj_directory
        self.participant_dir_list = [os.path.join(self.proj_dir, f'p{str(i).zfill(2)}') for i in
                                     range(self.start_participant, self.end_participant + 1)]
        self.camera_views = ['CAM_LL', 'CAM_LR']
        self.track_confidence = track_confidence
        self.detect_confidence = detect_confidence

    def extract_landmarks(self):
        total_count_landmark = 0
        for participant_dir in tqdm(self.participant_dir_list, desc="Participant Progress"):
            participant_dir_str = os.path.basename(participant_dir)
            logger.info(f'Processing participant {participant_dir_str}',
                        extra={'task_name': 'Landmark Extraction', 'detail': 'Participant Processing'})