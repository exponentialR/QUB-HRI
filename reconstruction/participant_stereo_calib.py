import os
from stereo_calibration import StereoCalibration
import cv2
import argparse

if __name__ == '__main__':
    # use argparse for start participant number and end participant number
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, help='Start participant number')
    parser.add_argument('--end', type=int, help='End participant number')
    args = parser.parse_args()

    PROJECT_DIR = '/home/iamshri/Documents/Dataset'
    START_PARTICIPANT = args.start
    END_PARTICIPANT = args.end
    for participant_id in range(START_PARTICIPANT, END_PARTICIPANT+1):
        parent_dir = os.path.join(PROJECT_DIR, f'p{participant_id:02d}')
        left_calib_path = os.path.join(parent_dir, 'CAM_LL/calib_param_CALIBRATION.npz')
        right_calib_path = os.path.join(parent_dir, 'CAM_LR/calib_param_CALIBRATION.npz')
        left_video_path = os.path.join(parent_dir, 'CAM_LL/calibration.mp4')
        right_video_path = os.path.join(parent_dir, 'CAM_LR/calibration.mp4')
        calib_prefix = (left_video_path.split('_')[1].split('/')[0].lower() + '_' + \
                       right_video_path.split('_')[1].split('/')[
                           0].lower()).upper()

        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)
        left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Left video frame count: {left_frame_count}')
        print(f'Right video frame count: {right_frame_count}')

        stereo_calib = StereoCalibration(left_calib_path, right_calib_path, left_video_path, right_video_path,
                                         frame_interval=5, min_corners=10, calib_prefix=calib_prefix)
        stereo_data_path = stereo_calib.run()
        print(f'Stereo calibration data saved to {stereo_data_path}')

    print('Stereo Calibration of participant ', f'p{START_PARTICIPANT:02d}', 'to', f'p{END_PARTICIPANT:02d}', 'is done!')

