import os
import cv2
import numpy as np
import mediapipe as mp

from utils import get_video_fps, load_calibration_data


class Video_EFGHU:
    def __init__(self, video_path, calib_path, max_hands=2):
        self.video_path = video_path
        self.calib_path = calib_path
        self.cam_matrix, self.dist_coeffs, _, _ = load_calibration_data(calib_path)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_tracking_confidence=.5, min_detection_confidence=.5,
                                                         refine_landmarks=True)
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5,
                                              max_num_hands=max_hands)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=.5, min_tracking_confidence=.5)

        self.head_pose_indices = [33, 263, 1, 61, 291, 199]
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]
        self.fps = get_video_fps(video_path)

    def process_HGF(self, frame, frame_index):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        timestamp = frame_index / self.fps


