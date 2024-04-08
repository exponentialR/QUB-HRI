import cv2
from reconstruction.downgrade_fps import downgrade_fps, match_frame_length
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def synchronize_videos(reference_video_path, target_video_path):
    downgrade_fps(reference_video_path, target_video_path)
    _, _ = match_frame_length(reference_video_path, target_video_path)


def get_video_files(directory, cam_view):
    cam_dir = os.path.join(directory, cam_view)
    return [f for f in os.listdir(cam_dir) if f.endswith(('.mp4', '.avi'))]


class VideoSynchronizer:
    def __init__(self, base_dir, start_participant, end_participant):
        self.base_dir = base_dir
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.reference_view = 'CAM_LR'
        self.target_views = ['CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV']

    def synchronize(self):
        for part_id in range(self.start_participant, self.end_participant + 1):
            participant_dir = os.path.join(self.base_dir, f'p{part_id:02d}')
            reference_video_files = get_video_files(participant_dir, self.reference_view)
            logger.info(f"Starting synchronization for participant p{part_id:02d}")

            for video_name in tqdm(reference_video_files, desc=f'Processing participant {part_id:02d}'):
                reference_video_path = os.path.join(participant_dir, self.reference_view, video_name)

                for target_view in self.target_views:
                    target_video_path = os.path.join(participant_dir, target_view, video_name)
                    if os.path.exists(target_video_path):
                        logger.info(f'Synchronizing {reference_video_path} and {target_video_path}')
                        synchronize_videos(reference_video_path, target_video_path)
                    else:
                        logger.warning(f'Video {target_video_path} does not exist. Skipping...')
