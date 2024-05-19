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

    def match_pixel_et_compute_disparity(self):
        face_disparties = self.right_keypoints['face_landmarks'][:, :, 0] - self.left_keypoints['face_landmarks'][:, :, 0]
        pose_disparities = self.right_keypoints['pose_landmarks'][:, :, 0] - self.left_keypoints['pose_landmarks'][:, :, 0]
        left_hand_disparities = self.right_keypoints['left_hand_landmarks'][:, :, 0] - self.left_keypoints['left_hand_landmarks'][:, :, 0]
        right_hand_disparities = self.right_keypoints['right_hand_landmarks'][:, :, 0] - self.left_keypoints['right_hand_landmarks'][:, :, 0]
        self.disparities = {'face_landmarks': face_disparties, 'pose_landmarks': pose_disparities,
                            'left_hand_landmarks': left_hand_disparities,
                            'right_hand_landmarks': right_hand_disparities}
        print('Disparities computed')

    def calculate_depth(self):
        f = self.stereo_calib['P1'][0, 0]
        self.depths = {}
        for key, disparity in self.disparities.items():
            depths = f * self.baseline / (disparity + 1e-6)
            self.depths[key] = depths
        print('Depths calculated')

    def convert_3D_coordinates(self):
        c_x, c_y = self.stereo_calib['P1'][0, 2], self.stereo_calib['P1'][1, 2]
        f = self.stereo_calib['P1'][0, 0]
        self.point_3D = {}
        for key, depths in self.depths.items():
            points_3D = []
            keypoints = self.left_keypoints[key]  # Use keypoints from the dataset
            for i in range(keypoints.shape[0]):
                frame_3D = []
                for j in range(keypoints.shape[1]):
                    x_L, y_L = keypoints[i, j, 0], keypoints[i, j, 1]
                    Z = depths[i, j]
                    X = (x_L - c_x) * Z / f
                    Y = (y_L - c_y) * Z / f
                    frame_3D.append([X, Y, Z])
                points_3D.append(frame_3D)
            self.point_3D[key] = np.array(points_3D)
        print('3D coordinates computed.')

    def transform_to_common_coord(self):
        R = self.stereo_calib['R']
        T = self.stereo_calib['T']
        self.points_3D_world = {}
        for key, points_3D_sets in self.point_3D.items():
            points_3D_world_sets = []
            for points_3D in points_3D_sets:
                points_3D_world = []
                for frame_3D in points_3D:
                    frame_3D_world = [np.dot(R, point) + T for point in frame_3D]
                    points_3D_world.append(frame_3D_world)
                points_3D_world_sets.append(points_3D_world)
            self.points_3D_world[key] = np.array(points_3D_world_sets)
        print('Transformed to common coordinate system.')

    def save_3D_coordinates(self):
        with h5py.File(self.output_hdf5, 'w') as f:
            for key, points_3D_world in self.points_3D_world.items():
                f.create_dataset(key, data=points_3D_world)
        print(f'3D coordinates saved to {self.output_hdf5}')

    def run(self):
        self.compute_baseline()
        self.match_pixel_et_compute_disparity()
        self.calculate_depth()
        self.convert_3D_coordinates()
        self.transform_to_common_coord()
        self.save_3D_coordinates()


if __name__ == '__main__':
    left_hdf5_path = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BS.hdf5'
    right_hdf5_path = '/home/iamshri/Documents/Dataset/p01/CAM_LR/landmarks/BIAH_BS.hdf5'
    output_hdf5_path = 'output.hdf5'

    stereo_calib = '/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz'
    triangulate = Triangulation(left_hdf5_path, right_hdf5_path, output_hdf5_path, stereo_calib)
    print(triangulate.left_frame_count)
    print(triangulate.right_frame_count)
    triangulate.run()
