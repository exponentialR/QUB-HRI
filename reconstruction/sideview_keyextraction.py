"""
Landmarks Extraction and Storage Module

This module processes video data to extract facial, pose, and hand landmarks using MediaPipe and saves them in an HDF5 format.
It supports high accuracy landmark detection and is tailored for research and development in fields requiring detailed human interaction analysis.

License:
Copyright (C) 2024 S Adebayo
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Dependencies:
- cv2 (opencv-python)
- mediapipe
- h5py
- numpy
- tqdm

Usage:
- Define the video path and the output path for the HDF5 file.
- Adjust the detection and tracking confidence thresholds as needed.
- Run the module to process the video and store the landmarks.

Author: Samuel Adebayo
Year: 2024

"""

import cv2
import mediapipe as mp
import h5py
import numpy as np
from tqdm import tqdm


def convert_to_pixel_coords(norm_x, norm_y, frame_width, frame_height):
    """
    Convert normalized coordinates to pixel coordinates.
    Args:
        norm_x: normalized x coordinate from MediaPipe via corrected video
        norm_y: normalized y coordinate from MediaPipe via corrected video
        frame_width: width of the frame, used to convert normalized coordinates to pixel coordinates
        frame_height: height of the frame, used to convert normalized coordinates to pixel coordinates

    Returns:
        tuple: pixel_x, pixel_y

    """
    pixel_x = int(norm_x * frame_width)
    pixel_y = int(norm_y * frame_height)
    return pixel_x, pixel_y


class LandmarksToHDF5:
    def __init__(self, vid_path, landmark_hdf5_path, detect_confidence=0.3, track_confidence=0.3, logger=None):
        self.video_path = vid_path
        self.hdf5_path = landmark_hdf5_path
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=detect_confidence,
                                                         min_tracking_confidence=track_confidence)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=detect_confidence,
                                           min_tracking_confidence=track_confidence)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                              min_detection_confidence=detect_confidence,
                                              min_tracking_confidence=track_confidence)
        self.logger = logger

    def process_and_save(self):
        """
        Process the video and save the landmarks to an HDF5 file.
        The process is as follows:
        - Open the video file
        - Get the frame count
        - Create HDF5 datasets to store face, pose and hand landmarks
        - Process each frame and store the landmarks in the respective datasets
        - Close the video file and the HDF5 file
        Returns:

        """
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

                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = self.face_mesh.process(frame_rgb)
                    pose_results = self.pose.process(frame_rgb)
                    hand_results = self.hands.process(frame_rgb)

                    # Store face landmarks
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            for idx, landmark in enumerate(face_landmarks.landmark):
                                px, py = convert_to_pixel_coords(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
                                face_dset[frame_idx, idx] = (px, py)
                    else:
                        face_dset[frame_idx] = np.zeros((478, 2))

                    # Store pose landmarks
                    if pose_results.pose_landmarks:
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            px, py = convert_to_pixel_coords(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
                            pose_dset[frame_idx, idx] = (px, py)
                    else:
                        pose_dset[frame_idx] = np.zeros((33, 2))

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks, classification in zip(hand_results.multi_hand_landmarks,
                                                                  hand_results.multi_handedness):
                            if classification.classification[0].label == 'Left':
                                hand_dataset = left_hand_dset
                            else:
                                hand_dataset = right_hand_dset
                            for idx, landmark in enumerate(hand_landmarks.landmark):
                                px, py = convert_to_pixel_coords(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
                                hand_dataset[frame_idx, idx] = (px, py)
                    frame_idx += 1
                    pbar.update(1)

        cap.release()
        print(f"Landmarks saved to HDF5 : {self.hdf5_path}")


if __name__ == '__main__':
    # video_path = "/home/iamshri/Documents/Dataset/Test_Evironment/p03/CAM_LL/BIAH_RB.mp4"  # Replace with the path to your video file
    # video_path ='/home/iamshri/Documents/Dataset/p01/CAM_LR/BIAH_BV.mp4'
    video_path = '/home/iamshri/Downloads/BIAH_RB.mp4'
    hdf5_path = 'landmarks_data.hdf5'  # Path where you want to save the HDF5 file
    # Get the dimension of the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video Dimensions: {width} x {height}")

    processor = LandmarksToHDF5(video_path, hdf5_path)
    processor.process_and_save()
