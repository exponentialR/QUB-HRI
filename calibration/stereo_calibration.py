import os.path

import numpy as np
import cv2
from helper_utils import extract_synchronized_frames, stereo_calibrate

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_SAME_FOCAL_LENGTH


class StereoCalibration:
    def __init__(self, left_calibration_data, right_calibration_data, left_video_path, right_video_path):
        self.stereo_data_path = os.path.join(os.path.dirname(os.path.dirname(left_calibration_data)),
                                             'stereo_calibration_params.npz')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
        self.left_calibration_data, self.right_calibration_data = left_calibration_data, right_calibration_data
        self.left_video_path, self.right_video_path = left_video_path, right_video_path

    def save_stereo_data(self, stereo_data):
        np.savez(self.stereo_data_path, **stereo_data)
        print(f'Stereo calibration data saved to {self.stereo_data_path}')

    def run(self):
        extracted_frames_left, extracted_frames_right = extract_synchronized_frames(self.left_video_path,
                                                                                    self.right_video_path,
                                                                                    self.aruco_dict, self.board)
        stereo_data = stereo_calibrate(self.left_calibration_data, self.right_calibration_data, extracted_frames_left,
                                       extracted_frames_right, termination_criteria=TERMINATION_CRITERIA, flags=FLAGS)
        self.save_stereo_data(stereo_data)
