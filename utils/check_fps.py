import cv2
import os
import csv
from calibration.data_specific_calib import is_empty
from tqdm import tqdm

def singleCalibMultiCorrect():
    """
    Perform single calibration and multiple corrections.

    This function goes through each participant directory and calibrates based on the 'CALIBRATION.MP4' video.
    It then performs corrections on the other videos in the same directory.
    """
    os.makedirs('csvlogs') if not os.path.exists('csvlogs') else None
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    with open(f'csvlogs/frame_rate.csv', mode='a', newline='') as c_file:
        frame_rate_writer = csv.writer(c_file)
        if is_empty('csvlogs/frame_rate.csv'):
            frame_rate_writer.writerow(["ParticipantID", "CameraViews", "video", "fps"])

        # Create or open a log file to log directories with more than 10 videos
        with open(f'csvlogs/differ_fps.csv', mode='a', newline='') as v_file:
            # Write headers to log files
            video_stats_writer = csv.writer(v_file)
            if is_empty('csvlogs/video_stats.csv'):
                video_stats_writer.writerow(["ParticipantID", "CameraViews", "video", "fps"])

            for participant_id in tqdm(sorted(self.participant_list), desc="Processing participants", position=0,
                                       leave=False):
                if not os.path.exists(participant_id):
                    self.logger.info(f'{participant_id} not exist')
                    pass
                else:
                    for camera_view in tqdm(os.listdir(participant_id), desc='Processing camera views', position=0,
                                            leave=False):
                        current_camera = os.path.join(participant_id, camera_view)

                        # Perform Calibration

                        calib_video_file = os.path.join(current_camera, 'CALIBRATION.MP4')
                        self.video_parent_dir = os.path.dirname(calib_video_file)
                        calib_file = f"{self.video_parent_dir}/{self.save_path_prefix}_{os.path.basename(calib_video_file).split('.')[0]}.npz"
                        if not os.path.exists(calib_file):
                            calib_file_path = self.calibrate_single_video(calib_video_file)
                            pass