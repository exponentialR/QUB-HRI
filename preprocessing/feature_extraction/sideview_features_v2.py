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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocessing.landmark_extraction import setup_calibration_video_logger

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

logger = setup_calibration_video_logger(
    "Subtask-Sideview-Features-Logger",
    format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %('
               'message)s',
    extra_attrs=['task_name', 'detail'],
    error_log_file='landmark_extraction_log.txt',
    levels_to_save={logging.DEBUG, logging.INFO},
    console_level=logging.INFO
)


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

                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            for idx, landmark in enumerate(face_landmarks.landmark):
                                px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, frame.shape[1],
                                                                      frame.shape[0])
                                face_dset[frame_idx, idx] = (px, py)
                    else:
                        face_dset[frame_idx] = np.zeros((478, 2))

                    if pose_results.pose_landmarks:
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, frame.shape[1],
                                                                  frame.shape[0])
                            pose_dset[frame_idx, idx] = (px, py)
                    else:
                        pose_dset[frame_idx] = np.zeros((33, 2))

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks, classification in zip(hand_results.multi_hand_landmarks,
                                                                  hand_results.multi_handedness):
                            hand_dataset = left_hand_dset if classification.classification[
                                                                 0].label == 'Left' else right_hand_dset
                            for idx, landmark in enumerate(hand_landmarks.landmark):
                                px, py = self.convert_to_pixel_coords(landmark.x, landmark.y, frame.shape[1],
                                                                      frame.shape[0])
                                hand_dataset[frame_idx, idx] = (px, py)

                    frame_idx += 1
                    pbar.update(1)
                    del frame, face_results, pose_results, hand_results  # Explicitly delete to free memory
                    gc.collect()  # Collect garbage

        cap.release()
        gc.collect()  # Final garbage collection
        if self.logger:
            self.logger.info(f"Landmarks saved to HDF5 : {self.hdf5_path}")

    @staticmethod
    def convert_to_pixel_coords(x, y, width, height):
        return int(x * width), int(y * height)


def process_video_task(args):
    video_path, output_path = args
    landmark_extractor = LandmarksToHDF5(video_path, output_path, logger=logger)
    landmark_extractor.process_and_save()


class SubtaskSideviewFeatures:
    def __init__(self, proj_dir, output_dir, camera_pairs='CAM_LL_CAM_LR', num_processes=4):
        self.proj_dir = proj_dir
        self.output_dir = output_dir
        self.camera_pairs = CAMERA_PAIRS[camera_pairs]
        self.num_processes = num_processes

    def extract_landmarks(self):
        subtask_list = [folder for folder in os.listdir(self.proj_dir) if os.path.isdir(os.path.join(self.proj_dir, folder))]
        for subtask in subtask_list:
            subtask_dir = os.path.join(self.proj_dir, subtask)
            current_subtask_files = [file for file in os.listdir(subtask_dir) if file.endswith('.mp4') and file.split('-')[1].startswith(self.camera_pairs)]
            os.makedirs(os.path.join(self.output_dir, subtask), exist_ok=True)
            tasks = [(os.path.join(subtask_dir, file), os.path.join(self.output_dir, subtask, file.replace('.mp4', '.h5'))) for file in current_subtask_files]

            with Pool(processes=self.num_processes) as pool:
                pool.map(process_video_task, tasks)
                pool.close()
                pool.join()
            logger.info(f'Processed {len(current_subtask_files)} videos in subtask {subtask}')



if __name__ == "__main__":
    proj_dir = '/home/samueladebayo/Documents/PhD/QUB-PHEO-Dataset/test-annotate'
    output_dir = '/media/samueladebayo/EXTERNAL_USB/landmark'
    subtask_sideview_features = SubtaskSideviewFeatures(proj_dir, output_dir)
    subtask_sideview_features.extract_landmarks()
