import os.path

import numpy as np
import cv2
from helper_utils import extract_synchronized_frames, extrinsic_calibration
from transfer_calibration import copy_calibration_videos

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_SAME_FOCAL_LENGTH
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD_USED = cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                    ARUCO_DICT)


class ExtrinsicCalibration:
    def __init__(self, left_calibration_data, right_calibration_data, left_video_path, right_video_path):
        self.stereo_data_path = os.path.join(os.path.dirname(os.path.dirname(left_calibration_data)),
                                             'extrinsic_calib.npz')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
        self.left_calibration_data, self.right_calibration_data = left_calibration_data, right_calibration_data
        self.left_video_path, self.right_video_path = left_video_path, right_video_path

    def save_stereo_data(self, stereo_data):
        print(f'stereo_data: {stereo_data}')
        np.savez(self.stereo_data_path, **stereo_data)
        print(f'Stereo calibration data saved to {self.stereo_data_path}')
        return self.stereo_data_path

    # def extrinsic_calibration(self, left_calibration_data, right_calibration_data, left_video_path, right_video_path):

    def run(self):
        # Transfer Original Calibration Videos:
        # Copy the original calibration videos from the original project directory to the new project directory

        extracted_frames_left, extracted_frames_right = extract_synchronized_frames(self.left_video_path,
                                                                                    self.right_video_path,
                                                                                    self.aruco_dict, self.board)
        extracted_frames_left_paths = [os.path.join(extracted_frames_left, files) for files in
                                       sorted(os.listdir(extracted_frames_left)) if
                                       files.endswith('.png')]
        extracted_frames_right_paths = [os.path.join(extracted_frames_right, files) for files in
                                        sorted(os.listdir(extracted_frames_right)) if
                                        files.endswith('.png')]
        print(f'Extracted {len(extracted_frames_left_paths)} frames from {self.left_video_path}')
        print(f'Extracted {len(extracted_frames_right_paths)} frames from {self.right_video_path}')
        stereo_image_pairs = list(zip(extracted_frames_left_paths, extracted_frames_right_paths))
        # stereo_data = extrinsic_calibration(stereo_image_pairs, self.left_calibration_data, self.right_calibration_data, BOARD_USED)
        stereo_data = extrinsic_calibration(self.left_calibration_data, self.right_calibration_data, stereo_image_pairs,
                                            BOARD_USED, termination_criteria=TERMINATION_CRITERIA,
                                            flags=FLAGS)
        self.save_stereo_data(stereo_data)
        return self.stereo_data_path
