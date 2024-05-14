import cv2
import h5py
import numpy as np
import os
import logging
from tqdm import tqdm
import h5py

class TriangulateViews:
    def __init__(self, l_hdf5_path, r_hdf5_path, o_hdf5_path, logger=None, stereo_calib=None):
        self.left_hdf5_path = l_hdf5_path
        self.right_hdf5_path = r_hdf5_path
        self.output_hdf5_path = o_hdf5_path
        self.logger = logger
        self.datasets = ['face_landmarks', 'pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']
        if stereo_calib is not None:    # Load the stereo calibration data
            self.steeo_calib = np.load(stereo_calib)

            # Load Claibration data
            self.T = self.stereo_data['T']
            self.R = self.stereo_data['R']

    def load_keypoints(self, hdf_path):
        with h5py.File(hdf_path, 'r') as f:
            keypoints = {key: f[key][:] for key in f.keys()}
        return keypoints

    def load_selected_datasets(self, file_path):
        hdf5_file = h5py.File(file_path, 'r')
        selected_data = {}
        for dataset in self.datasets:
            if dataset in hdf5_file.keys():
                selected_data[dataset] = hdf5_file[dataset]
            else:
                self.logger.error(f'Dataset {dataset} not found in {file_path}')
        return selected_data

    def compute_disparity(self):
        pass

    def calculate_depth(self):
        pass

    def convert_3D_coordinates(self):
        pass

    def transform_coordinates(self):
        pass

    def triangulate(self):
        # load the left hdf5 file
        left_hdf5_file = h5py.File(self.left_hdf5_path, 'r')
        right_hdf5_file = h5py.File(self.right_hdf5_path, 'r')
        # Ensure the keypoints have same length in both files
        assert len(left_hdf5_file.keys()) == len(right_hdf5_file.keys()), "Mismatch in number of keypoints."
        print(f'Number of keypoints: {len(left_hdf5_file.keys())}')

        # for key, value in left_hdf5_file.items():
        #     left_data_shape = value.shape
        #     right_data_shape = right_hdf5_file[key].shape
        #     if left_data_shape != right_data_shape:
        #         self.logger.error(f'Shape mismatch between left and right datasets for {key}')
        #         return
        #     self.logger.info(f'Shape of {key} dataset: {left_data_shape}')
        # print(f'Frame Length: {left_data_shape[0]}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    left_hdf5_path = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BS.hdf5'
    right_hdf5_path = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BS.hdf5'
    output_hdf5_path = 'output.hdf5'

    triangulator = TriangulateViews(left_hdf5_path, right_hdf5_path, output_hdf5_path, logger)
    triangulator.triangulate()
