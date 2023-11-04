import cv2
import os
import csv
from calibration.data_specific_calib import is_empty
from tqdm import tqdm
from calibration.logger_utils import setup_calibration_video_logger

logger = setup_calibration_video_logger(logger_name='Check-Video-Integrity')


class CHECK_VID_INTEGRITY:
    def __init__(self, proj_repo, participant_id_start, participant_id_last):
        self.proj_repo = proj_repo
        self.participant_id_start = participant_id_last
        self.participant_id_start = participant_id_last
        self.participant_list = [os.path.join(proj_repo, f'p{participant_id:02d}') for participant_id in
                                 range(participant_id_start, participant_id_last + 1)]
        self.video_extensions = ('.avi', '.mp4', '.mov', '.mkv')

    def check_integrity(self):
        """
        Checks for conflicting FPS in videos of same participant
        """
        os.makedirs('csvlogs') if not os.path.exists('csvlogs') else None
        # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        with open(f'csvlogs/frame_rate.csv', mode='a', newline='') as c_file:
            frame_rate_writer = csv.writer(c_file)
            if is_empty('csvlogs/frame_rate.csv'):
                frame_rate_writer.writerow(
                    ["ParticipantID", "CameraViews", "Video", "FPS", "FrameCount", "WIDTH", "HEIGHT"])

            # Create or open a log file to log directories with more than 10 videos
            with open(f'csvlogs/differ_fps.csv', mode='a', newline='') as v_file:
                # Write headers to log files
                video_stats_writer = csv.writer(v_file)
                if is_empty('csvlogs/differ_fps.csv'):
                    video_stats_writer.writerow(
                        ["ParticipantID", "CameraViews", "Video", "FPS", "FrameCount", "WIDTH", "HEIGHT"])

                for participant_id in tqdm(sorted(self.participant_list), desc="Processing participants", position=0,
                                           leave=False):
                    if not os.path.exists(participant_id):
                        logger.info(f'{participant_id} not exist')
                        pass
                    else:
                        for camera_view in tqdm(os.listdir(participant_id), desc='Processing camera views', position=0,
                                                leave=False):
                            current_camera = os.path.join(participant_id, camera_view)

                            # Perform Calibration

                            calib_video_file = os.path.join(current_camera, 'CALIBRATION.MP4')
                            video_parent_dir = os.path.dirname(calib_video_file)
                            videos_camera_view = os.listdir(current_camera)
                            videos_camera_view = [i for i in videos_camera_view if
                                                  i.lower().endswith(self.video_extensions)]
                            # Perform Correction
                            if len(videos_camera_view) <= 10:
                                total_vids_len = len(videos_camera_view)
                                checked_video_count = 0
                                camera_view_fps = []
                                for idx, video_file in enumerate(videos_camera_view):
                                    video_file_path = os.path.join(current_camera, video_file)
                                    integrity_details = self.video_integrity(video_file_path)


    def video_integrity(self, video_file_path):
        """
        Checks for a video frame per seconds, width, and height

        Parameters:
        - video_file_path (str): The path to the video file to be processed.

        Returns:
        - str: The path to the corrected video.
        """

        video_name = os.path.basename(video_file_path)
        video_name_ = os.path.basename(video_file_path).split('.')[0]

        cap = cv2.VideoCapture(video_file_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print(f'Original frame width x height: {frame_width} x {frame_height}')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger(
            f"The FPS of the video {video_name_} is {fps} with width x height: {frame_width}x{frame_height} and frame count: {frame_count}")
        integrity_details = {'fps': fps, 'width': frame_width, 'height': frame_height, 'count': frame_count}
        return integrity_details
