from stereo_calibration import StereoCalibration
import logging
import os

START_PARTICIPANT = 1
END_PARTICIPANT = 1
HOME_DIR = '/home/qub-hri/Documents/QUBVisionData/RawData/stereo'

# Assuming CAMERA_PAIRS is correctly defined as before
CAMERA_PAIRS = [
    ('CAM_LL', 'CAM_LR'),
    ('CAM_UL', 'CAM_UR'),
    ('CAM_UR', 'CAM_UL'),
    ('CAM_UL', 'CAM_LL'),
    ('CAM_AV', 'CAM_LL'),
    ('CAM_AV', 'CAM_LR'),
    ('CAM_AV', 'CAM_UL'),
    ('CAM_AV', 'CAM_UR')
]


def perform_stereo_pair_calib():
    for participant_id in range(START_PARTICIPANT, END_PARTICIPANT+1):
        participant_dir = os.path.join(HOME_DIR, f'P{participant_id:02d}')

        for left_cam, right_cam in CAMERA_PAIRS:
            left_calib_path = os.path.join(participant_dir, left_cam, 'calib_param_CALIBRATION.npz')
            right_calib_path = os.path.join(participant_dir, right_cam, 'calib_param_CALIBRATION.npz')
            left_video_path = os.path.join(participant_dir, left_cam, 'CALIBRATION_CC.mp4')
            right_video_path = os.path.join(participant_dir, right_cam, 'CALIBRATION_CC.mp4')

            if not all([os.path.exists(left_calib_path), os.path.exists(right_calib_path), os.path.exists(left_video_path), os.path.exists(right_video_path)]):
                logging.warning(f'Not all required files exist for participant {participant_id}, skipping')
                continue

            calib_prefix = left_cam.split('_')[1].lower() + right_cam.split('_')[1].lower()

            try:
                stereo_calib = StereoCalibration(left_calib_path, right_calib_path, left_video_path, right_video_path, calib_prefix, frame_interval=1, min_corners=15)
                stereo_data_path = stereo_calib.run()
                logging.info(f'Stereo calibration data saved to {stereo_data_path}')
            except Exception as e:
                logging.error(f'Error occurred while processing participant {participant_id}, camera pair {left_cam} and {right_cam}: {e}')
                continue
    return