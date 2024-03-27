import json
import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

from utils import get_video_fps, extract_landmarks


class ExtractLandmark:
    def __init__(self, image, cam_matrix, dist_coeffs, path_h5=None, max_hands=2):
        self.image = image
        self.path_hd5 = path_h5
        self.cam_matrix, self.dist_coeffs = cam_matrix, dist_coeffs
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_tracking_confidence=.5, min_detection_confidence=.5,
                                                         refine_landmarks=True)
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5,
                                              max_num_hands=max_hands)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=.5, min_tracking_confidence=.5)

        self.head_pose_indices = [33, 263, 1, 61, 291, 199]
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image

    def calculate_head_pose(self, processed_image, face_landmarks):
        """
        Given an image and corresponding facial landmark,
        this method extracts head pose [pitch, yaw, and roll], Iris in 3D and 2D, and forehead in 3D and 2D
        :param processed_image: Mediapipe processed image
        :param face_landmarks: 473 facial landmarks
        :return:
        """
        processed_image_height, processed_image_width, _ = processed_image.shape
        face, face_2d = [], []
        left_iris, right_iris = None, None
        forehead = None

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in self.head_pose_indices:
                x, y = int(lm.x * processed_image_width), int(lm.y * processed_image_height)
                face.append([x, y, lm.z])
                face_2d.append([x, y])
            if idx == 6:
                forehead = (lm.x * processed_image_width, lm.y * processed_image_height, int(lm.z))
            if idx == 468:
                left_iris = (int(lm.x * processed_image_width), int(lm.y * processed_image_height), int(lm.z))
            if idx == 473:
                right_iris = (int(lm.x * processed_image_width), int(lm.y * processed_image_height), int(lm.z))

        face_extracted = np.array(face, dtype=np.float64)
        face_2d = np.array(face, dtype=np.float64)

        if self.cam_matrix is not None and self.dist_coeffs is not None:
            cam_matrix = self.cam_matrix
            dist_matrix = self.dist_coeffs
        else:
            focal_length = 1 * processed_image_width
            cam_matrix = np.array([[focal_length, 0, processed_image_height / 2],
                                   [0, focal_length, processed_image_width / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(face_extracted, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rvec)
        euler_angles = cv2.decomposeProjectionMatrix(rmat)[6]
        pitch, yaw, roll = euler_angles
        text = 'HeadEyeData'
        head_eye = {
            'head_pitch': pitch, 'head_yaw': yaw, 'head_roll': roll, 'forehead': forehead,
            'rvec': rvec, 'tvec': tvec,
            'cam_matrix': cam_matrix,
            'dist_matrix': dist_matrix,
            'left_iris': left_iris, 'right_iris': right_iris,
            'text': text
        }
        return head_eye

