import numpy as np
import h5py
import cv2
import plotly.graph_objects as go


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

    def visualize_3d_animation(self, start_frame, end_frame):
        frames = []
        for frame_number in range(start_frame, end_frame + 1):
            points_3d_face, points_3d_left_hand, points_3d_right_hand, points_3d_pose = self.get_3d_points_for_frame(
                frame_number)
            frame_data = [
                go.Scatter3d(
                    x=points_3d_face[:, 0],
                    y=points_3d_face[:, 1],
                    z=points_3d_face[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='red',
                        opacity=0.8,
                        colorbar=dict(title='Face')
                    ),
                    name='Face'
                ),
                go.Scatter3d(
                    x=points_3d_left_hand[:, 0],
                    y=points_3d_left_hand[:, 1],
                    z=points_3d_left_hand[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='blue',
                        opacity=0.8,
                        colorbar=dict(title='Left Hand')
                    ),
                    name='Left Hand'
                ),
                go.Scatter3d(
                    x=points_3d_right_hand[:, 0],
                    y=points_3d_right_hand[:, 1],
                    z=points_3d_right_hand[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='green',
                        opacity=0.8,
                        colorbar=dict(title='Right Hand')
                    ),
                    name='Right Hand'
                ),
                go.Scatter3d(
                    x=points_3d_pose[:, 0],
                    y=points_3d_pose[:, 1],
                    z=points_3d_pose[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='orange',
                        opacity=0.8,
                        colorbar=dict(title='Pose')
                    ),
                    name='Pose'
                )
            ]
            frames.append(go.Frame(data=frame_data, name=str(frame_number)))

        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(title='X axis'),
                    yaxis=dict(title='Y axis'),
                    zaxis=dict(title='Z axis')
                ),
                title='3D Keypoints Animation',
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": 100, "redraw": True},
                                               "fromcurrent": True, "mode": "immediate"}])]
                )]
            ),
            frames=frames
        )

        fig.show()


# Paths to the files
left_hdf5_file = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BV.hdf5'
right_hdf5_file = '/home/iamshri/Documents/Dataset/p01/CAM_LR/landmarks/BIAH_BV.hdf5'
calib_file = '/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz'
# Instantiate the class

# Instantiate the class
stereo_reconstruction = Stereo3DReconstruction(calib_file, left_hdf5_file, right_hdf5_file)

# Visualize the 3D points animation from frame 300 to 310
start_frame = 214
end_frame = 923
stereo_reconstruction.visualize_3d_animation(start_frame, end_frame)
