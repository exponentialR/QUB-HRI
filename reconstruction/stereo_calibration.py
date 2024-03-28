"""
Stereo calibration module

This module performs stereo calibration using Charuco boards for synchronized stereo video pairs.
It Detects the Charuco corners, and calculates the extrinsic parameters of the stereo camera pair.

License:
Copyright (C) 2024 QUB
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Dependencies:
- numpy
- opencv-python
- opencv-contrib-python
- tqdm

Usage:
- Ensure that the left and right camera calibration data and video paths are correctly defined.
- Ensure you have the necessary calibration data for each camera and video files for both left and right cameras

Author: Samuel Adebayo
Year: 2024

"""

import os.path

import numpy as np
import cv2
from extrinsic_calibration import extrinsic_calibration
from extract_sync_frames import ExtractSyncFrames

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5  # 48.2 reprojection error
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
BOARD_USED = cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                    ARUCO_DICT)


class StereoCalibration:
    """
    Class to perform stereo calibration using Charuco boards for synchronized stereo video pairs.

    Attributes:
    - left_calibration_data: Path to the npz file containing left camera calibration data.
    - right_calibration_data: Path to the npz file containing right camera calibration data.
    - left_video_path: Path to the left video file.
    - right_video_path: Path to the right video file.
    - calib_prefix: Prefix used for saving the stereo calibration data.
    - min_corners: Minimum number of corners to detect in each frame.
    - frame_interval: Interval for extracting frames from the video.
    - max_frames: Maximum number of frames to extract from the video.
    - aruco_dict: The Aruco dictionary used for calibration.
    - board: The Charuco board used for calibration.

    """

    def __init__(self, left_calibration_data, right_calibration_data, left_video_path, right_video_path, calib_prefix,
                 min_corners=2, frame_interval=1, max_frames=1000):
        self.stereo_data_path = os.path.join(os.path.dirname(os.path.dirname(left_calibration_data)),
                                             f'{calib_prefix}stereo_calib.npz')
        self.min_corners = min_corners
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
        self.left_calibration_data, self.right_calibration_data = left_calibration_data, right_calibration_data
        self.left_video_path, self.right_video_path = left_video_path, right_video_path
        self.frame_interval = frame_interval
        self.max_frames = max_frames

    def save_stereo_data(self, stereo_data):
        """
        Saves the stereo calibration data to a file.

        Parameters:
        stereo_data (dict): A dictionary containing the stereo calibration data.
        """
        print(f'stereo_data: {stereo_data}')
        np.savez(self.stereo_data_path, **stereo_data)
        print(f'Stereo calibration data saved to {self.stereo_data_path}')
        return self.stereo_data_path

    def run(self):
        """
        Executes the stereo calibration process.

        :return: str: The path to the saved stereo calibration data.
        """
        extracted_frames_left, extracted_frames_right = ExtractSyncFrames(self.left_video_path, self.right_video_path,
                                                                          self.aruco_dict, self.board,
                                                                          self.frame_interval, self.min_corners,
                                                                          self.max_frames).extract_synchronized_frames()

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
    left_calib_path = '/media/qub-hri/Seagate/Test_Evironment/p03/CAM_LL/calib_param_CALIBRATION.npz'
    right_calib_path = '/media/qub-hri/Seagate/Test_Evironment/p03/CAM_LR/calib_param_CALIBRATION.npz'
    left_video_path = '/media/qub-hri/Seagate/Test_Evironment/p03/CAM_LL/CALIBRATION_CC.MP4'
    right_video_path = '/media/qub-hri/Seagate/Test_Evironment/p03/CAM_LR/CALIBRATION_CC.MP4'
    calib_prefix = left_video_path.split('_')[1].split('/')[0].lower() + right_video_path.split('_')[1].split('/')[
        0].lower()

    left_cap = cv2.VideoCapture(left_video_path)
    right_cap = cv2.VideoCapture(right_video_path)
    left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Left video frame count: {left_frame_count}')
    print(f'Right video frame count: {right_frame_count}')

    stereo_calib = StereoCalibration(left_calib_path, right_calib_path, left_video_path, right_video_path,
                                     frame_interval=1, min_corners=15, calib_prefix=calib_prefix)
    stereo_data_path = stereo_calib.run()
    print(f'Stereo calibration data saved to {stereo_data_path}')
    pass
