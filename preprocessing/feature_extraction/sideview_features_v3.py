import os
import sys
import cv2
import h5py
import numpy as np
import logging
from tqdm import tqdm
import mediapipe as mp
import gc  # Garbage collector interface
from multiprocessing import Pool
import threading
from queue import Queue

# Assuming these modules are defined appropriately
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocessing.landmark_extraction import setup_calibration_video_logger

# Setup logger
logger = setup_calibration_video_logger(
    "Subtask-Sideview-Features-Logger",
    format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %(message)s',
    extra_attrs=['task_name', 'detail'],
    error_log_file='landmark_extraction_log.txt',
    levels_to_save={logging.DEBUG, logging.INFO},
    console_level=logging.INFO
)

# Define camera pairs
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

        with h5py.File(self.hdf5_path, 'w') as f:
            face_dset = f.create_dataset("face_landmarks", (frame_count, 478, 2), dtype='f')
            pose_dset = f.create_dataset("pose_landmarks", (frame_count, 33, 2), dtype='f')
            left_hand_dset = f.create_dataset("left_hand_landmarks", (frame_count, 21, 2), dtype='f')
            right_hand_dset = f.create_dataset("right_hand_landmarks", (frame_count, 21, 2), dtype='f')

            with tqdm(total=frame_count, desc=f"Processing Video {self.video_path}") as pbar:
                frame_idx = 0
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = self.face_mesh.process(frame_rgb)
                    pose_results = self.pose.process(frame_rgb)
                    hand_results = self.hands.process(frame_rgb)

                    self.handle_face_landmarks(face_results, face_dset, frame_idx, frame.shape[1], frame.shape[0])
                    self.handle_pose_landmarks(pose_results, pose_dset, frame_idx, frame.shape[1], frame.shape[0])
                    self.handle_hand_landmarks(hand_results, left_hand_dset, right_hand_dset, frame_idx, frame.shape[1],
                                               frame.shape[0])

                    frame_idx += 1
                    pbar.update(1)
                    del frame, face_results, pose_results, hand_results  # Explicitly delete to free memory
                    gc.collect()  # Collect garbage

        cap.release()
        gc.collect()  # Final garbage collectio

    @staticmethod
    def convert_to_pixel_coords(x, y, width, height):
        """Convert normalized coordinates to pixel coordinates."""
        return int(x * width), int(y * height)

    def handle_face_landmarks(self, results, dataset, frame_idx, width, height):
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(landmarks.landmark):
                    px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, width, height)
                    dataset[frame_idx, idx] = (px, py)
        else:
            dataset[frame_idx] = np.zeros((478, 2))

    def handle_pose_landmarks(self, results, dataset, frame_idx, width, height):
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, width, height)
                dataset[frame_idx, idx] = (px, py)
        else:
            dataset[frame_idx] = np.zeros((33, 2))

    def handle_hand_landmarks(self, results, left_dataset, right_dataset, frame_idx, width, height):
        """Process hand landmarks and store them in the corresponding datasets."""
        if results.multi_hand_landmarks:
            for hand_landmarks, classification in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_dataset = left_dataset if classification.classification[0].label == 'Left' else right_dataset
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, width, height)
                    hand_dataset[frame_idx, idx] = (px, py)
        else:
            left_dataset[frame_idx] = np.zeros((21, 2))  # Assuming all hands have 21 landmarks
            right_dataset[frame_idx] = np.zeros((21, 2))


def process_video_task(args):
    video_path, output_path = args
    landmark_extractor = LandmarksToHDF5(video_path, output_path, logger=logger)
    landmark_extractor.process_and_save()


def worker(queue):
    while True:
        args = queue.get()
        if args is None:  # None is the signal to stop.
            break
        process_video_task(args)
        queue.task_done()


class SubtaskSideviewFeatures:
    def __init__(self, proj_dir, output_dir, camera_pairs='CAM_LL_CAM_LR', num_threads=8, batch_size=8):
        self.proj_dir = proj_dir
        self.output_dir = output_dir
        self.camera_pairs = CAMERA_PAIRS[camera_pairs]
        self.num_threads = num_threads
        self.batch_size = batch_size
    def extract_landmarks(self):
        queue = Queue()
        threads = [threading.Thread(target=worker, args=(queue,)) for _ in range(self.num_threads)]

        for thread in threads:
            thread.start()

        subtask_list = [folder for folder in os.listdir(self.proj_dir) if
                        os.path.isdir(os.path.join(self.proj_dir, folder))]
        for subtask in subtask_list:
            subtask_dir = os.path.join(self.proj_dir, subtask)
            current_subtask_files = [file for file in os.listdir(subtask_dir) if
                                     file.endswith('.mp4') and file.split('-')[1].startswith(self.camera_pairs)]
            os.makedirs(os.path.join(self.output_dir, subtask), exist_ok=True)

            for i in range(0, len(current_subtask_files), self.batch_size):
                batch_files = current_subtask_files[i:i + self.batch_size]
                tasks = [(os.path.join(subtask_dir, file),
                          os.path.join(self.output_dir, subtask, file.replace('.mp4', '.h5'))) for file in batch_files]
                for task in tasks:
                    queue.put(task)
                logger.info(
                    f'Processed batch {i // self.batch_size + 1} in subtask {subtask}, total {len(batch_files)} files.')

        queue.join()


if __name__ == "__main__":
    proj_dir = '/home/samueladebayo/Documents/PhD/QUB-PHEO-Dataset/test-annotate'
    output_dir = '/media/samueladebayo/EXTERNAL_USB/landmark'
    subtask_sideview_features = SubtaskSideviewFeatures(proj_dir, output_dir)
    subtask_sideview_features.extract_landmarks()
