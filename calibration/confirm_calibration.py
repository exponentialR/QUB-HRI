import os.path
import os
import numpy as np
import cv2
from tqdm import tqdm


def check_calibration_param_shape(calib_path):
    calib_data = np.load(calib_path)
    mtx, dist, rvecs, tvecs = calib_data['mtx'], calib_data['dist'], calib_data['rvecs'], calib_data['tvecs']
    mtx_shape, dist_shape, rvecs_shape, tvecs_shape = mtx.shape, dist.shape, rvecs.shape, tvecs.shape
    # print(mtx_shape, dist_shape, rvecs_shape, tvecs_shape)
    return mtx_shape, dist_shape, rvecs_shape, tvecs_shape


def attempt_recalibration(calib_path):
    mtx_shape, dist_shape, rvecs_shape, tvecs_shape = check_calibration_param_shape(calib_path)
    if mtx_shape != (3, 3) or dist_shape != (1, 5) or rvecs_shape != (1, 3) or tvecs_shape != (1, 3):
        print('Invalid calibration parameters. Attempting recalibration...')
        # Implement your recalibration logic here
        # You can use the OpenCV calibration functions to recalibrate the camera
        # and save the new calibration data to new_calib_path
        pass
    else:
        print('Calibration parameters are valid')


if __name__ == '__main__':
    calib_path = '/media/iamshri/Seagate/QUB-PHEOVision/p01/CAM_LR/calib_param_CALIBRATION.npz'

    new_calib_path = 'path/to/new_calibration_data.npz'
    attempt_recalibration(calib_path)
