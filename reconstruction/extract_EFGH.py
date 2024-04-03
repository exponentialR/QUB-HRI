import json

import numpy as np

from frame_EFGH import FrameEFGH
from downgrade_fps import downgrade_fps, match_frame_length
import cv2
import os
import h5py
from tqdm import tqdm
from utils import load_calibration_data, create_dir, save_to_hdf5, convert_to_list


class Sync_DataExtraction:
    def __init__(self, left_video_path, right_video_path, frame_interval=1, max_frames=1000):
        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        self.left_video_name = os.path.basename(left_video_path)[0:-4]
        self.right_video_name = os.path.basename(right_video_path)[0:-4]
        self.right_dir = create_dir(os.path.join(os.path.dirname(right_video_path), 'ProcessedData'))
        self.left_dir = create_dir(os.path.join(os.path.dirname(left_video_path), 'ProcessedData'))

        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.left_cam_matrix, self.left_dist_coeffs, _, _ = load_calibration_data(self.left_video_path)
        self.right_cam_matrix, self.right_dist_coeffs, _, _ = load_calibration_data(self.right_video_path)
        self.left_fps = None
        self.right_fps = None

    def sync_data_extract(self):
        # downgraded_video = downgrade_fps(self.left_video_path, self.right_video_path)
        # if downgraded_video:
        #     _, _ = match_frame_length(self.left_video_path, self.right_video_path)

        left_hdf5_path = os.path.join(self.left_dir, f'{self.left_video_name}.hdf5')
        right_hdf5_path = os.path.join(self.right_dir, f'{self.right_video_name}.hdf5')

        left_cap = cv2.VideoCapture(self.left_video_path)
        right_cap = cv2.VideoCapture(self.right_video_path)
        if self.left_fps is None or self.right_fps is None:
            self.left_fps = left_cap.get(cv2.CAP_PROP_FPS)
            self.right_fps = right_cap.get(cv2.CAP_PROP_FPS)

        left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count_difference = abs(left_frame_count - right_frame_count)
        skipped = None
        max_frame = min(left_frame_count, right_frame_count)
        if left_frame_count > right_frame_count:
            for _ in range(frame_count_difference):
                skipped = 'left'
                left_cap.read()

        elif right_frame_count > left_frame_count:
            for _ in range(frame_count_difference):
                right_cap.read()
                skipped = 'right'

        else:
            print('Videos are already synchronized')
        print(f'{frame_count_difference} frames skipped in {skipped} video')

        with h5py.File(left_hdf5_path, 'w') as left_hdf_file, h5py.File(right_hdf5_path, 'w') as right_hdf_file:
            pbar = tqdm(total=max_frame, desc='Processing Frames and Extracting Data from Left Video', unit='frames')
            right_timestamps, right_frame_indices, right_face_presence, right_gaze_vector, right_face_landmarks, right_hand_landmarks, right_upper_body_pose, = [], [], [], [], [], [], []
            left_timestamps, left_frame_indices, left_face_presence, left_gaze_vector, left_face_landmarks, left_hand_landmarks, left_upper_body_pose = [], [], [], [], [], [], []

            head_eye_data_left = {key: [] for key in
                                  ['head_pitch', 'head_yaw', 'head_roll', 'left_iris', 'right_iris', 'forehead', 'rvec',
                                   'tvec', 'cam_matrix', 'dist_matrix', 'text']}
            head_eye_data_right = {key: [] for key in
                                   ['head_pitch', 'head_yaw', 'head_roll', 'left_iris', 'right_iris', 'forehead', 'rvec',
                                    'tvec', 'cam_matrix', 'dist_matrix', 'text']}
            processed_frame = 0
            while left_cap.isOpened() and right_cap.isOpened() and processed_frame < max_frame:
                left_time = left_cap.get(cv2.CAP_PROP_POS_MSEC)
                right_time = right_cap.get(cv2.CAP_PROP_POS_MSEC)
                time_difference = abs(left_time - right_time)
                if time_difference > (1000 / max_frame):
                    if left_time < right_time:
                        left_cap.read()
                    else:
                        right_cap.read()
                    continue
                ret_left, left_frame = left_cap.read()
                ret_right, right_frame = right_cap.read()
                if not ret_left or not ret_right:
                    print(
                        f'Done processing videos {os.path.basename(self.left_video_path)} and {os.path.basename(self.right_video_path)}')
                    break

                if processed_frame % self.frame_interval == 0:
                    try:

                        left_efgh = FrameEFGH(left_frame, processed_frame, self.left_cam_matrix, self.left_dist_coeffs,
                                              fps=self.left_fps)
                        # right_efgh = FrameEFGH(right_frame, processed_frame, self.right_cam_matrix,
                        #                        self.right_dist_coeffs,
                        #                        fps=self.right_fps)
                        left_data = left_efgh.processEFGH()
                        # right_data = right_efgh.processEFGH()

                        left_timestamps.append(left_data['timestamp'])
                        left_frame_indices.append(left_data['frame_index'])
                        left_face_presence.append(left_data['face_presence'])
                        left_gaze_vector.append(left_data['gaze_vector'])
                        left_face_landmarks.append(left_data['facial_landmarks'])
                        left_hand_landmarks.append(left_data['hand_landmarks'])
                        left_upper_body_pose.append(left_data['upper_body_pose'])

                        # right_timestamps.append(right_data['timestamp'])
                        # right_frame_indices.append(right_data['frame_index'])
                        # right_face_presence.append(right_data['face_presence'])
                        # right_gaze_vector.append(right_data['gaze_vector'])
                        # right_face_landmarks.append(right_data['facial_landmarks'])
                        # right_hand_landmarks.append(right_data['hand_landmarks'])
                        # right_upper_body_pose.append(right_data['upper_body_pose'])
                        #
                        # for key in head_eye_data_left:
                        #     head_eye_data_left[key].append(left_data['head_eye_data'][key])
                        # for key in head_eye_data_right:
                        #     head_eye_data_right[key].append(right_data['head_eye_data'][key])

                    except Exception as e:
                        print(f'Error processing frame {processed_frame} due to {e}')
                        break
                    pbar.update(1)
                    processed_frame += 1

            # Batch save to hdf5

            left_hdf_file.create_dataset('timestamps', data=np.array(left_timestamps))
            left_hdf_file.create_dataset('frame_indices', data=np.array(left_frame_indices))
            left_hdf_file.create_dataset('face_presence', data=np.array(left_face_presence))
            left_hdf_file.create_dataset('gaze_vector', data=np.array(left_gaze_vector))
            left_hdf_file.create_dataset('facial_landmarks', data=np.array(left_face_landmarks))
            left_hdf_file.create_dataset('hand_landmarks', data=np.array(left_hand_landmarks))
            left_hdf_file.create_dataset('upper_body_pose', data=np.array(left_upper_body_pose))
            for key, values in head_eye_data_left.items():
                left_hdf_file.create_dataset(f'head_eye_data/{key}', data=np.array(values))
            #
            # right_hdf_file.create_dataset('timestamps', data=np.array(right_timestamps))
            # right_hdf_file.create_dataset('frame_indices', data=np.array(right_frame_indices))
            # right_hdf_file.create_dataset('face_presence', data=np.array(right_face_presence))
            # right_hdf_file.create_dataset('gaze_vector', data=np.array(right_gaze_vector))
            # right_hdf_file.create_dataset('facial_landmarks', data=np.array(right_face_landmarks))
            # right_hdf_file.create_dataset('hand_landmarks', data=np.array(right_hand_landmarks))
            # right_hdf_file.create_dataset('upper_body_pose', data=np.array(right_upper_body_pose))
            # for key, values in head_eye_data_right.items():
            #     right_hdf_file.create_dataset(f'head_eye_data/{key}', data=np.array(values))
        pbar.close()


if __name__ == '__main__':
    left_video_path = '/home/iamshri/Documents/Dataset/Test_Evironment/p03/CAM_LL/BIAH_RB.mp4'
    right_video_path = '/home/iamshri/Documents/Dataset/Test_Evironment/p03/CAM_LR/BIAH_RB.mp4'
    synch_data = Sync_DataExtraction(left_video_path, right_video_path)
    synch_data.sync_data_extract()
    print('Data extraction complete')
