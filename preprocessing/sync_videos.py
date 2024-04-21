import time

import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import logging
from tqdm import tqdm
import subprocess
import re
from downgrade_fps import downgrade_fps, match_frame_length
from logger_utils import setup_calibration_video_logger
import shutil


def synchronize_videos(reference_video_path, target_video_path, logger):
    downgrade_fps(reference_video_path, target_video_path, logger=logger)
    _, _ = match_frame_length(reference_video_path, target_video_path, logger=logger)
    return reference_video_path, target_video_path


def get_video_files(directory, cam_view):
    cam_dir = os.path.join(directory, cam_view)
    return [f for f in os.listdir(cam_dir) if f.endswith(('.mp4', '.avi'))]


class VideoSynchronizer:
    def __init__(self, base_dir, start_participant, end_participant):
        self.base_dir = base_dir
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.reference_view = 'CAM_LR'
        self.target_views = sorted(['CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV'])
        self.logger = setup_calibration_video_logger(
            "Video-Synchronization-Logger",
            format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %('
                       'message)s',
            extra_attrs=['task_name', 'detail'],
            error_log_file='sync_videos_log.txt',
            levels_to_save={logging.DEBUG, logging.INFO},  # Set levels to save
            console_level = logging.INFO  # Set console level
        )

    def synchronize(self):
        for part_id in range(self.start_participant, self.end_participant + 1):
            participant_dir = os.path.join(self.base_dir, f'p{part_id:02d}')
            reference_video_files = get_video_files(participant_dir, self.reference_view)
            for video_name in tqdm(reference_video_files,
                                   desc=f'Processing participant p{part_id:02d}'):
                reference_video_path = os.path.join(participant_dir, self.reference_view, video_name)
                for target_view in self.target_views:
                    target_video_path = os.path.join(participant_dir, target_view, video_name)
                    
                    if os.path.exists(target_video_path):
                        if f'p{part_id:02d}' == 'p42' and video_name == 'TOWER_MS.mp4':
                            
                            unsync_path = os.path.join(participant_dir, 'unsyn_videos', target_view)
                            
                            os.makedirs(unsync_path, exist_ok=True) if not os.path.exists(unsync_path) else None
                            unsync_video = os.path.join(unsync_path, video_name)
                            shutil.move(reference_video_path, unsync_video)
                            self.logger.debug(f'UNSYNC VIDEO MOVED TO {unsync_lr_video}. MOVING', extra={'task_name': f'MOVING UNSYNCHRONISED {video_name}', 'detail':'Video moved'})
                            if target_view == 'CAM_LR':
                                unsyc_lr = os.path.join(participant_dir, 'unsync_videos', 'CAM_LR')
                                os.makedirs(unsyc_lr, exist_ok=True) if not os.path.exists(unsyc_lr) else None
                                unsync_lr_video = os.path(unsyc_lr, video_name)
                                shutil.move(reference_video_path, unsyc_lr)
                                self.logger.debug(f'UNSYNC VIDEO MOVED TO {unsync_lr_video}. MOVING', extra={'task_name': f'MOVING UNSYNCHRONISED {video_name}', 'detail':'Video moved'})

                            print(f'{unsync_video}: TOWER_MS.mp4 Found')
                            # continue
                        else:
                            print ('will process this')
                            # synchronize_videos(reference_video_path, target_video_path, logger=self.logger)
                    else:
                        self.logger.warning(f'Video {target_video_path} does not exist. Skipping...', extra={'task_name': 'Synchronization', 'detail': 'Video does not exist'})

if __name__ == '__main__':
    base_dir = '/media/BlueHDD/QUB-PHEO-datasets'
    start_participant = 42
    end_participant = 42
    synchronizer = VideoSynchronizer(base_dir, start_participant, end_participant)
    synchronizer.synchronize()
