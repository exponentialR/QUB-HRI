import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Stereo3DReconstruction:
    def __init__(self, calib_file, left_hdf5_file, right_hdf5_file):
        # Load the stereo calibration data
        self.stereo_calib = np.load(calib_file)
        self.camera_matrix_left = self.stereo_calib['left_mtx']
        self.dist_coeffs_left = self.stereo_calib['left_dist']
        self.camera_matrix_right = self.stereo_calib['right_mtx']
        self.dist_coeffs_right = self.stereo_calib['right_dist']
        self.R = self.stereo_calib['R']
        self.T = self.stereo_calib['T']

        # Load keypoint data for the left view
        self.face_landmarks_left, self.left_hand_landmarks_left, self.pose_landmarks_left, self.right_hand_landmarks_left = self.load_keypoints(
            left_hdf5_file)

        # Load keypoint data for the right view
        self.face_landmarks_right, self.left_hand_landmarks_right, self.pose_landmarks_right, self.right_hand_landmarks_right = self.load_keypoints(
            right_hdf5_file)

        # Projection matrices
        self.P1 = np.dot(self.camera_matrix_left, np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.P2 = np.dot(self.camera_matrix_right, np.hstack((self.R, self.T.reshape(3, 1))))

    def load_keypoints(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            face_landmarks = f['face_landmarks'][:]  # shape (923, 478, 2)
            left_hand_landmarks = f['left_hand_landmarks'][:]  # shape (923, 21, 2)
            pose_landmarks = f['pose_landmarks'][:]  # shape (923, 33, 2)
            right_hand_landmarks = f['right_hand_landmarks'][:]  # shape (923, 21, 2)
        return face_landmarks, left_hand_landmarks, pose_landmarks, right_hand_landmarks

    def triangulate_points(self, pts_left, pts_right):
        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, pts_left, pts_right)
        points_4d = points_4d_hom / points_4d_hom[3]
        return points_4d[:3].T

    def get_3d_points_for_frame(self, frame_number):
        pts_left = self.face_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right = self.face_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d = self.triangulate_points(pts_left, pts_right)
        return points_3d

    def visualize_3d_points(self, points_3d):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap='viridis',
                        marker='o')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        fig.colorbar(sc, ax=ax, label='Z-coordinate')
        ax.set_title('3D Keypoints Visualization')
        plt.show()


# Paths to the files
left_hdf5_file = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BV.hdf5'
right_hdf5_file = '/home/iamshri/Documents/Dataset/p01/CAM_LR/landmarks/BIAH_BV.hdf5'
calib_file = '/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz'

# Instantiate the class
stereo_reconstruction = Stereo3DReconstruction(calib_file, left_hdf5_file, right_hdf5_file)

# Get and visualize 3D points for frame 303
frame_number = 303  # Frame number to visualize
points_3d = stereo_reconstruction.get_3d_points_for_frame(frame_number)

# Visualize the 3D points using Matplotlib
stereo_reconstruction.visualize_3d_points(points_3d)
