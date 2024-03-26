import os.path

import numpy as np
import cv2
from helper_utils import extract_synchronized_frames, extrinsic_calibration
from transfer_calibration import copy_calibration_videos

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 # 48.2 reprojection error
# FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_INTRINSIC # 20.2 reprojection error
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # with first flag 48.2, with second flag 20.2
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) # (CAM_LL, CAM_LR) 0.9038792595999602 reprojection error with first flag and 2.3419323982344147 with second
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
BOARD_USED = cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                    ARUCO_DICT)


class ExtrinsicCalibration:
    def __init__(self, left_calibration_data, right_calibration_data, left_video_path, right_video_path, min_corners=2, frame_interval=1, max_frames=1000):
        self.stereo_data_path = os.path.join(os.path.dirname(os.path.dirname(left_calibration_data)),
                                             'stereo_calib.npz')
        self.min_corners = min_corners
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
        self.left_calibration_data, self.right_calibration_data = left_calibration_data, right_calibration_data
        self.left_video_path, self.right_video_path = left_video_path, right_video_path
        self.frame_interval = frame_interval
        self.max_frames = max_frames

    def save_stereo_data(self, stereo_data):
        print(f'stereo_data: {stereo_data}')
        np.savez(self.stereo_data_path, **stereo_data)
        print(f'Stereo calibration data saved to {self.stereo_data_path}')
        return self.stereo_data_path

    def run(self):
        extracted_frames_left, extracted_frames_right = extract_synchronized_frames(self.left_video_path,
                                                                                    self.right_video_path,
                                                                                    self.aruco_dict, self.board,
                                                                                    frame_interval=self.frame_interval,
                                                                                    min_corners=self.min_corners, max_frames=self.max_frames)
        extracted_frames_left_paths = [os.path.join(extracted_frames_left, files) for files in
                                       sorted(os.listdir(extracted_frames_left)) if
                                       files.endswith('.png')]
        extracted_frames_right_paths = [os.path.join(extracted_frames_right, files) for files in
                                        sorted(os.listdir(extracted_frames_right)) if
                                        files.endswith('.png')]
        print(f'Extracted {len(extracted_frames_left_paths)} frames from {self.left_video_path}')
        print(f'Extracted {len(extracted_frames_right_paths)} frames from {self.right_video_path}')
        stereo_image_pairs = list(zip(extracted_frames_left_paths, extracted_frames_right_paths))
        stereo_data = extrinsic_calibration(self.left_calibration_data, self.right_calibration_data, stereo_image_pairs,
                                            BOARD_USED, termination_criteria=TERMINATION_CRITERIA,
                                            flags=FLAGS)
        self.save_stereo_data(stereo_data)
        return self.stereo_data_path


if __name__ == '__main__':
    left_calib_path = '/home/iamshri/Documents/PHEO-Data/Test-ground/p07/CAM_UL/calib_param_CALIBRATION.npz'
    right_calib_path = '/home/iamshri/Documents/PHEO-Data/Test-ground/p07/CAM_UR/calib_param_CALIBRATION.npz'
    left_video_path = '/home/iamshri/Documents/PHEO-Data/Test-ground/p07/CAM_UL/CALIBRATION_CC.MP4'
    right_video_path = '/home/iamshri/Documents/PHEO-Data/Test-ground/p07/CAM_UR/CALIBRATION_CC.MP4'
    left_cap = cv2.VideoCapture(left_video_path)
    right_cap = cv2.VideoCapture(right_video_path)
    left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Left video frame count: {left_frame_count}')
    print(f'Right video frame count: {right_frame_count}')
    extrinsic_calib = ExtrinsicCalibration(left_calib_path, right_calib_path, left_video_path, right_video_path, frame_interval=1, min_corners=15)
    stereo_data_path = extrinsic_calib.run()
    print(f'Stereo calibration data saved to {stereo_data_path}')
    pass
