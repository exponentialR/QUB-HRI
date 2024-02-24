import cv2
import os
import numpy as np
from confirm_calibration import check_calibration_param_shape
from helper_utils import calibrate_single_video


class MatchCalibParams:
    def __init__(self, calib_path_1, calib_path_2, video_path_1, video_path_2, pattern_size=(16, 11), square_size=33,
                 marker_size=26, dictionary='DICT_4X4_100'):
        self.calib_path_1, self.calib_path_2 = calib_path_1, calib_path_2
        self.video_path_1, self.video_path_2 = video_path_1, video_path_2
        self.calibration_video_path_1 = os.path.join(os.path.dirname(self.video_path_1), 'Original_CALIBRATION.MP4')
        self.calibration_video_path_2 = os.path.join(os.path.dirname(self.video_path_2), 'Original_CALIBRATION.MP4')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        self.board = cv2.aruco.CharucoBoard(pattern_size, square_size / 1000, marker_size / 1000,
                                            self.aruco_dict)
        self.frame_interval_calib = 10
        self.save_path_prefix = "reclib"
        self.min_corners = 15

    def check_calibration_param_shape(self):
        mtx_shape_1, dist_shape_1, rvecs_shape_1, tvecs_shape_1 = check_calibration_param_shape(self.calib_path_1)
        mtx_shape_2, dist_shape_2, rvecs_shape_2, tvecs_shape_2 = check_calibration_param_shape(self.calib_path_2)
        if mtx_shape_1 == mtx_shape_2 and dist_shape_1 == dist_shape_2 and rvecs_shape_1 == rvecs_shape_2 and tvecs_shape_1 == tvecs_shape_2:
            check = {'check': True, 'message': 'Calibration parameters match'}
            return check
        else:
            check = {'check': False, 'message': 'Calibration parameters do not match',
                     'calib_1': (mtx_shape_1, dist_shape_1, rvecs_shape_1, tvecs_shape_1),
                     'calib_2': (mtx_shape_2, dist_shape_2, rvecs_shape_2, tvecs_shape_2)}
            return check

    def run(self):
        check = self.check_calibration_param_shape()
        if check['check'] == True:
            print(check['message'])
            return
        else:
            print(check['message'])
            print(f'Calibration 1: {check["calib_1"]}')
            print(f'Calibration 2: {check["calib_2"]}')
            self.calib_path_1, self.calib_path_2 = None, None
            min_shape = min(check['calib_1'][2][0], check['calib_2'][2][0])
            recalib_path_1, recalib_path_2 = self.stereo_camera_calibration(73)
            self.calib_path_1, self.calib_path_2 = recalib_path_1, recalib_path_2
            check = self.check_calibration_param_shape()
            if not check['check']:
                new_min_shape = min(check['calib_1'][2][0], check['calib_2'][2][0])
                print('Recalibration failed')
                print(f'Attempting to Recalibrate')
                recalib_path_1, recalib_path_2 = self.stereo_camera_calibration(new_min_shape)
                return recalib_path_1, recalib_path_2
            else:
                print('Recalibration successful')
                return recalib_path_1, recalib_path_2

    def stereo_camera_calibration(self, max_calib_frames):
        print(f'Now doing Recalibration')

        calib_path_1 = calibrate_single_video(self.calibration_video_path_1, self.aruco_dict, self.board, self.frame_interval_calib,
                                              min_corners=self.min_corners,
                                              max_calib_frames=max_calib_frames)
        calib_path_2 = calibrate_single_video(self.calibration_video_path_2, self.aruco_dict, self.board, self.frame_interval_calib,
                                              min_corners=self.min_corners,
                                              max_calib_frames=max_calib_frames)
        return calib_path_1, calib_path_2


if __name__ == '__main__':
    calib_path_1 = '/media/iamshri/Seagate/QUB-PHEOVision/p01/CAM_LR/calib_param_CALIBRATION.npz'
    calib_path_2 = '/media/iamshri/Seagate/QUB-PHEOVision/p01/CAM_LL/calib_param_CALIBRATION.npz'
    video_path_1 = '/media/iamshri/Seagate/QUB-PHEOVision/p01/CAM_LR/BIAH_BS.mp4'

    video_path_2 = '/media/iamshri/Seagate/QUB-PHEOVision/p01/CAM_LL/BIAH_BS.mp4'

    match_calib_params = MatchCalibParams(calib_path_1, calib_path_2, video_path_1, video_path_2)
    match_calib_params.run()
