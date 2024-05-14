import numpy as np
import cv2
import h5py


class Triangulation:
    def __init__(self, left_hdf5, right_hdf5, output_hdf5, stereo_calib):
        self.left_hdf5 = left_hdf5
        self.right_hdf5 = right_hdf5
        self.output_hdf5 = output_hdf5
        self.stereo_calib = np.load(stereo_calib)
        self.left_keypoints, self.left_frame_count = self.load_keypoints(self.left_hdf5)
        self.right_keypoints, self.right_frame_count = self.load_keypoints(self.right_hdf5)
        self.baseline = None
        self.compute_baseline()
        self.disparities = None
        self.depths = None
        self.point_3D = None
        self.points_3D_world = None

    @staticmethod
    def load_keypoints(hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            keypoints = {key: f[key][:] for key in f.keys() if key != 'frame_count'}
            frame_count = len(f['face_landmarks'] if 'face_landmarks' in f.keys() else f['pose_landmarks'])
        return keypoints, frame_count

    def compute_baseline(self):
        self.baseline = np.linalg.norm(self.stereo_calib['T'])
        print(f'Baseline: {self.baseline:.2f} meters')

    def run(self):
        pass


if __name__ == '__main__':
    left_hdf5_path = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BS.hdf5'
    right_hdf5_path = '/home/iamshri/Documents/Dataset/p01/CAM_LR/landmarks/BIAH_BS.hdf5'
    output_hdf5_path = 'output.hdf5'

    stereo_calib = '/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz'
    triangulate = Triangulation(left_hdf5_path, right_hdf5_path, output_hdf5_path, stereo_calib)
    print(triangulate.left_frame_count)
    print(triangulate.right_frame_count)

