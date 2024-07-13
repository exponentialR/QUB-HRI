import os
from stereo import StereoCalibration
import cv2
import argparse


def check_exists(video_path):
    if not os.path.exists(video_path):
        print(f'{video_path} Does not exist!')
    else:
        pass


if __name__ == '__main__':
    # use argparse for start participant number and end participant number
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, help='Start participant number', default=1)
    parser.add_argument('--end', type=int, help='End participant number', default=5)
    parser.add_argument('--pair', type=str, help='Camera pair to calibrate', default='CAM_AV_CAM_LR')
    args = parser.parse_args()
    CAMERA_PAIRS = {
        'CAM_LL_CAM_LR': ('CAM_LL', 'CAM_LR'),
        'CAM_UL_CAM_UR': ('CAM_UL', 'CAM_UR'),
        'CAM_UR_CAM_UL': ('CAM_UR', 'CAM_UL'),
        'CAM_UL_CAM_LL': ('CAM_UL', 'CAM_LL'),
        'CAM_AV_CAM_LL': ('CAM_AV', 'CAM_LL'),
        'CAM_AV_CAM_LR': ('CAM_AV', 'CAM_LR'),
        'CAM_AV_CAM_UL': ('CAM_AV', 'CAM_UL'),
        'CAM_AV_CAM_UR': ('CAM_AV', 'CAM_UR')
    }
    PROJECT_DIR = '/media/samueladebayo/EXTERNAL_USB/QUB-PHEO-Proceesed'

    START_PARTICIPANT = args.start
    END_PARTICIPANT = args.end
    no_calib = 0
    yes_calib = 0
    left, right = CAMERA_PAIRS[args.pair]
    print(f'Starting stereo calibration from participant {START_PARTICIPANT} to participant {END_PARTICIPANT}')
    print(f'Camera pair: {left} and {right}')
    for participant_id in range(START_PARTICIPANT, END_PARTICIPANT + 1):
        if participant_id == 16:
            continue
        else:
            parent_dir = os.path.join(PROJECT_DIR, f'p{participant_id:02d}')
            left_calib_path = [os.path.join(parent_dir, left, file) for file in os.listdir(os.path.join(parent_dir, 'CAM_AV')) if
                 file.endswith('.npz')][0]
            right_calib_path = [os.path.join(parent_dir, right, file) for file in os.listdir(os.path.join(parent_dir, 'CAM_LR')) if
                 file.endswith('.npz')][0]
            left_video_path = os.path.join(parent_dir, f'{left}/calibration.mp4')
            right_video_path = os.path.join(parent_dir, f'{right}/calibration.mp4')
            calib_prefix = (os.path.dirname(left_video_path).split('_')[-1]) + '_' + (
                os.path.dirname(right_video_path).split('_')[-1])
            calib_prefix = f'p{participant_id:02d}-{calib_prefix}'

            calibration_file_name = (os.path.join(parent_dir, f'{calib_prefix}_stereo.npz'))
            if os.path.exists(calibration_file_name):
                print(calib_prefix, ' Already Exist')
                yes_calib += 1

            else:
                print(f'{calibration_file_name} Does not exist, Continuing to perform stereo calib now')
                no_calib += 1

                left_cap = cv2.VideoCapture(left_video_path)
                right_cap = cv2.VideoCapture(right_video_path)
                print(left_video_path)
                left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f'LEFT {left_video_path}: {left_frame_count}')
                print(f'RIGHT {right_video_path}: {right_frame_count}')
                frame_interval = 4
                min_corners = 10

                stereo_calib = StereoCalibration(left_calib_path, right_calib_path, left_video_path, right_video_path,
                                                 frame_interval=frame_interval, min_corners=min_corners,
                                                 calib_prefix=calib_prefix)
                stereo_data_path = stereo_calib.run()
                print(f'Stereo calibration data saved to {stereo_data_path}')
    print('Stereo Calibration of participant ', f'p{START_PARTICIPANT:02d}', 'to', f'p{END_PARTICIPANT:02d}',
          'is done!')

