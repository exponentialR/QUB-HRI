import os
import sys
import cv2
import h5py
import numpy as np
import logging
from tqdm import tqdm
import mediapipe as mp
import gc
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocessing.landmark_extraction import setup_calibration_video_logger

# Setup logging
# Ensure 'levels_to_save' parameter is passed
logger = setup_calibration_video_logger(
    "Subtask-Sideview-Features-Logger",
    format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %(message)s',
    extra_attrs=['task_name', 'detail'],
    error_log_file='landmark_extraction_log.txt',
    levels_to_save={logging.DEBUG, logging.INFO},
    console_level=logging.INFO
)


# Configuration for camera pairs
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


class LandmarksToHDF5:
    def __init__(self, vid_path, landmark_hdf5_path, detect_confidence=0.3, track_confidence=0.3, logger=None):
        self.video_path = vid_path
        self.hdf5_path = landmark_hdf5_path
        self.logger = logger
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=detect_confidence,
                                                         min_tracking_confidence=track_confidence)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=detect_confidence,
                                           min_tracking_confidence=track_confidence)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                              min_detection_confidence=detect_confidence,
                                              min_tracking_confidence=track_confidence)

    def process_and_save(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_size = 5  # Processing frames in batches of 10
        total_batches = (frame_count + batch_size - 1) // batch_size

        with h5py.File(self.hdf5_path, 'w') as f:
            face_dset = f.create_dataset("face_landmarks", (frame_count, 478, 2), dtype='f')
            pose_dset = f.create_dataset("pose_landmarks", (frame_count, 33, 2), dtype='f')
            left_hand_dset = f.create_dataset("left_hand_landmarks", (frame_count, 21, 2), dtype='f')
            right_hand_dset = f.create_dataset("right_hand_landmarks", (frame_count, 21, 2), dtype='f')

            with tqdm(total=frame_count, desc=f"Processing Video {self.video_path}") as pbar:
                frame_idx = 0
                for batch in range(total_batches):
                    for _ in range(batch_size):
                        success, frame = cap.read()
                        if not success:
                            break

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_results = self.face_mesh.process(frame_rgb)
                        pose_results = self.pose.process(frame_rgb)
                        hand_results = self.hands.process(frame_rgb)

                        # Process each type of landmark
                        self.process_landmarks(face_results, face_dset, frame_idx)
                        self.process_landmarks(pose_results, pose_dset, frame_idx, landmark_type='pose')
                        self.process_hand_landmarks(hand_results, left_hand_dset, right_hand_dset, frame_idx)

                        frame_idx += 1
                        pbar.update(1)

                    # Force garbage collection at the end of each batch
                    gc.collect()

        cap.release()
        if self.logger:
            self.logger.info(f"Landmarks saved to HDF5 : {self.hdf5_path}")

    def process_landmarks(self, results, dataset, frame_idx, landmark_type='face'):
        if landmark_type == 'pose' and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
        elif results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            dataset[frame_idx] = np.zeros((dataset.shape[1], 2))
            return

        for idx, landmark in enumerate(landmarks):
            px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, dataset.shape[2], dataset.shape[1])
            dataset[frame_idx, idx] = (px, py)

    def process_hand_landmarks(self, results, left_hand_dset, right_hand_dset, frame_idx):
        if not results.multi_hand_landmarks:
            left_hand_dset[frame_idx] = np.zeros((21, 2))
            right_hand_dset[frame_idx] = np.zeros((21, 2))
            return

        for hand_landmarks, classification in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_dataset = left_hand_dset if classification.classification[0].label == 'Left' else right_hand_dset
            for idx, landmark in enumerate(hand_landmarks.landmark):
                px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, hand_dataset.shape[2],
                                                      hand_dataset.shape[1])
                hand_dataset[frame_idx, idx] = (px, py)

    @staticmethod
    def convert_to_pixel_coords(x, y, width, height):
        return int(x * width), int(y * height)


def process_video_task(args):
    video_path, output_path = args
    landmark_extractor = LandmarksToHDF5(video_path, output_path, logger=logger)
    landmark_extractor.process_and_save()


class SubtaskSideviewFeatures:
    def __init__(self, proj_dir, output_dir, camera_pairs='CAM_LL_CAM_LR', num_processes=4, num_videos_to_process=50):
        self.proj_dir = proj_dir
        self.output_dir = output_dir
        self.camera_pairs = CAMERA_PAIRS[camera_pairs]
        self.num_processes = num_processes
        self.num_videos_to_process = num_videos_to_process

    def extract_landmarks(self):
        # List all video files and filter by camera pairs
        all_files = [file for file in os.listdir(self.proj_dir) if
                     file.endswith('.mp4') and file.split('-')[1].startswith(self.camera_pairs)]
        subtask = os.path.basename(self.proj_dir)
        os.makedirs(os.path.join(self.output_dir, subtask), exist_ok=True)

        # Prepare tasks but only for videos that don't already have a processed HDF5 file
        tasks = []
        processed_count = 0
        for file in all_files:
            output_hdf5 = os.path.join(self.output_dir, subtask, file.replace('.mp4', '.h5'))
            if not os.path.exists(output_hdf5):
                tasks.append((os.path.join(self.proj_dir, file), output_hdf5))
                processed_count += 1
                if processed_count >= self.num_videos_to_process:
                    break

        if tasks:
            with Pool(processes=self.num_processes) as pool:
                pool.map(process_video_task, tasks)
                pool.close()
                pool.join()
            logger.info(f'Processed {len(tasks)} videos in subtask {subtask}')
        else:
            logger.info('No new videos to process.')


if __name__ == "__main__":
    proj_dir = '/home/samueladebayo/Documents/PhD/QUB-PHEO-Dataset/test-annotate/BHO'
    output_dir = '/media/samueladebayo/EXTERNAL_USB/landmark'
    subtask_sideview_features = SubtaskSideviewFeatures(proj_dir, output_dir)
    subtask_sideview_features.extract_landmarks()
