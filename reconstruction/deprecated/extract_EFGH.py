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
        downgraded_video = downgrade_fps(self.left_video_path, self.right_video_path)
        if downgraded_video:
            _, _ = match_frame_length(self.left_video_path, self.right_video_path)

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
        left_data_list = []
        right_data_list = []

        with tqdm(total=max_frame, desc='Processing Frames and Extracting Data') as pbar:

            processed_frame = 0

            while processed_frame < max_frame and left_cap.isOpened() and right_cap.isOpened():
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
                    left_efgh = FrameEFGH(left_frame, processed_frame, self.left_cam_matrix, self.left_dist_coeffs, fps=self.left_fps)
                    right_efgh = FrameEFGH(right_frame, processed_frame, self.right_cam_matrix, self.right_dist_coeffs, fps=self.right_fps)
                    left_data = left_efgh.processEFGH()
                    right_data = right_efgh.processEFGH()
                    left_data_list.append(json.dumps(convert_to_list(left_data)))
                    right_data_list.append(json.dumps(convert_to_list(right_data)))
                    pbar.update(1)
                    processed_frame += 1

            # Batch save to hdf5
            save_to_hdf5(left_data_list, left_hdf5_path)
            save_to_hdf5(right_data_list, right_hdf5_path)


if __name__ == '__main__':
    left_video_path = '/media/iamshri/Seagate/Test_Evironment/p03/CAM_LL/BIAH_RB.mp4'
    right_video_path = '/media/iamshri/Seagate/Test_Evironment/p03/CAM_LR/BIAH_RB.mp4'
    synch_data = Sync_DataExtraction(left_video_path, right_video_path)
    synch_data.sync_data_extract()
    print('Data extraction complete')
