import numpy as np
import h5py
import cv2


def load_keypoints(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        face_landmarks = f['face_landmarks'][:]  # shape (923, 478, 2)
        left_hand_landmarks = f['left_hand_landmarks'][:]  # shape (923, 21, 2)
        pose_landmarks = f['pose_landmarks'][:]  # shape (923, 33, 2)
        right_hand_landmarks = f['right_hand_landmarks'][:]  # shape (923, 21, 2)
    return face_landmarks, left_hand_landmarks, pose_landmarks, right_hand_landmarks


class TriangulateViews:
    def __init__(self, stereo_path, left_path, right_path, output_file):
        self.stereo_calib = np.load(stereo_path)
        self.left_path = left_path
        self.right_path = right_path
        self.R = self.stereo_calib['R']
        self.T = self.stereo_calib['T']
        self.P1 = np.dot(self.stereo_calib['left_mtx'], np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.P2 = np.dot(self.stereo_calib['right_mtx'], np.hstack((self.R, self.T.reshape(3, 1))))
        self.face_landmarks_left, self.left_hand_landmarks_left, self.pose_landmarks_left, self.right_hand_landmarks_left = load_keypoints(
            left_path)
        self.face_landmarks_right, self.left_hand_landmarks_right, self.pose_landmarks_right, self.right_hand_landmarks_right = load_keypoints(
            right_path)
        self.output_file = output_file

    def triangulate_points(self, pts_left, pts_right):
        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, pts_left, pts_right)
        points_4d = points_4d_hom / points_4d_hom[3]
        return points_4d[:3].T  # Return only the 3D points

    def get_3d_points_for_frame(self, frame_number):
        pts_left_face = self.face_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_face = self.face_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_face = self.triangulate_points(pts_left_face, pts_right_face)

        pts_left_left_hand = self.left_hand_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_left_hand = self.left_hand_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_left_hand = self.triangulate_points(pts_left_left_hand, pts_right_left_hand)

        pts_left_right_hand = self.right_hand_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_right_hand = self.right_hand_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_right_hand = self.triangulate_points(pts_left_right_hand, pts_right_right_hand)

        pts_left_pose = self.pose_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_pose = self.pose_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_pose = self.triangulate_points(pts_left_pose, pts_right_pose)

        return points_3d_face, points_3d_left_hand, points_3d_right_hand, points_3d_pose

    def save_3d_points_hdf5(self):
        num_frames = self.face_landmarks_left.shape[0]
        with h5py.File(self.output_file, 'w') as f:
            face_points_3d = f.create_dataset('face_points_3d', (num_frames, 478, 3), dtype='f')
            left_hand_points_3d = f.create_dataset('left_hand_points_3d', (num_frames, 21, 3), dtype='f')
            right_hand_points_3d = f.create_dataset('right_hand_points_3d', (num_frames, 21, 3), dtype='f')
            pose_points_3d = f.create_dataset('pose_points_3d', (num_frames, 33, 3), dtype='f')

            for frame in range(num_frames):
                points_3d_face, points_3d_left_hand, points_3d_right_hand, points_3d_pose = self.get_3d_points_for_frame(
                    frame)
                face_points_3d[frame] = points_3d_face
                left_hand_points_3d[frame] = points_3d_left_hand
                right_hand_points_3d[frame] = points_3d_right_hand
                pose_points_3d[frame] = points_3d_pose

        print(f"3D points saved to {self.output_file}")


# Test the TriangulateViews class
if __name__ == '__main__':
    stereo_path = 'stereo_calib.npz'
    left_path = 'left_keypoints.hdf5'
    right_path = 'right_keypoints.hdf5'
    output_file = '3d_points.hdf5'

    triangulate_views = TriangulateViews(stereo_path, left_path, right_path, output_file)
    triangulate_views.save_3d_points_hdf5()
