import cv2
import os
import csv
from tqdm import tqdm
from calibration.logger_utils import setup_calibration_video_logger

logger = setup_calibration_video_logger(logger_name='Check-Video-Integrity')


def is_empty(filename):
    return os.path.getsize(filename) == 0


def log_exists(filename, row):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for existing_row in reader:
            if all(str(cell).strip() == str(value) for cell, value in zip(existing_row, row)):
                return True
    return False


class CheckVidIntegrity:
    def __init__(self, proj_repo, participant_id_start, participant_id_last):
        self.proj_repo = proj_repo
        self.participant_list = [os.path.join(proj_repo, f'p{participant_id:02d}') for participant_id in
                                 range(participant_id_start, participant_id_last + 1)]
        self.video_extensions = ('.avi', '.mp4', '.mov', '.mkv')

    def check_integrity(self):
        """
        Checks for conflicting FPS in videos of same participant
        """
        os.makedirs('csvlogs') if not os.path.exists('csvlogs') else None
        # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        with open(f'csvlogs/video_integrity.csv', mode='a', newline='') as c_file:
            video_integrity_writer = csv.writer(c_file)
            if is_empty('csvlogs/video_integrity.csv'):
                video_integrity_writer.writerow(
                    ["ParticipantID", "CameraViews", "Video", "FPS", "FrameCount", "WIDTH", "HEIGHT", 'DURATION'])

            # Create or open a log file to log directories with more than 10 videos
            with open(f'csvlogs/differ_integrity.csv', mode='a', newline='') as v_file:
                # Write headers to log files
                differ_integrity_writer = csv.writer(v_file)
                if is_empty('csvlogs/differ_integrity.csv'):
                    differ_integrity_writer.writerow(
                        ["ParticipantID", "CameraViews", "Video", "FPS", "FrameCount", "WIDTH", "HEIGHT", 'DURATION'])

                for participant_id in tqdm(sorted(self.participant_list), desc="Processing participants", position=0,
                                           leave=False):
                    if not os.path.exists(participant_id):
                        logger.info(f'{participant_id} not exist')
                        pass
                    else:
                        for camera_view in tqdm(os.listdir(participant_id),
                                                desc=f'Processing {os.path.basename(participant_id)} camera view {camera_view}',
                                                position=0,
                                                leave=False):

                            current_camera = os.path.join(participant_id, camera_view)
                            videos_camera_view = os.listdir(current_camera)
                            videos_camera_view = [i for i in videos_camera_view if
                                                  i.lower().endswith(self.video_extensions)]

                            if len(videos_camera_view) <= 10:
                                camera_view_fps = []
                                for idx, video_file in enumerate(videos_camera_view):
                                    video_file_path = os.path.join(current_camera, video_file)
                                    integrity_details = self.video_integrity(video_file_path)
                                    camera_view_fps.append(integrity_details['fps'])
                                    video_details = [participant_id, camera_view, video_file, integrity_details['fps'],
                                                     integrity_details['count'], integrity_details['width'],
                                                     integrity_details['height'], integrity_details['duration']]
                                    if len(set(camera_view_fps)) > 1:
                                        if not log_exists(f'csvlogs/differ_integrity.csv', video_details):
                                            differ_integrity_writer.writerow(video_details)
                                    if not log_exists(f'csvlogs/video_integrity.csv', video_details):
                                        video_integrity_writer.writerow(video_details)

    def compare_integrity(self, camera_view1='CAM_LL', camera_view2='CAM_LR'):
        for participant_id in tqdm(sorted(self.participant_list), desc="Processing participants", position=0,
                                   leave=False):
            if not os.path.exists(participant_id):
                logger.info(f'{participant_id} does not exist')
                pass
            else:
                camera_view1_path = os.path.join(participant_id, camera_view1)
                camera_view2_path = os.path.join(participant_id, camera_view2)

                if not os.path.exists(camera_view1_path) or not os.path.exists(camera_view2_path):
                    logger.info(f'One or both camera views do not exist for participant {participant_id}')
                    continue
                else:
                    fps_1, duration_1 = self.get_camera_fps_duration(camera_view1_path)
                    fps_2, duration_2 = self.get_camera_fps_duration(camera_view2_path)
                    if isinstance(fps_1, set) or isinstance(duration_1, set) or isinstance(fps_2, set) or isinstance(
                            duration_2, set):
                        logger.debug(
                            f'Manual review required for participant {participant_id} in camera views {camera_view1} and {camera_view2}')

                    else:

                        if fps_1 != fps_2:
                            logger.info(
                                f'FPS discrepancy found for participant {participant_id}, between {camera_view1} ({fps_1} fps) and {camera_view2} ({fps_2} fps)')
                            discrepancies_found = f'{camera_view1} : {fps_1}\n {camera_view2}:{fps_2}'
                        elif fps_1 != fps_2 and duration_1 != duration_2:
                            discrepancies_found = f'{camera_view1} : {fps_1} - Duration : {duration_1}\n {camera_view2}:{fps_2} - Duration : {duration_2}'
                        elif duration_1 != duration_2:
                            discrepancies_found = f'{camera_view1} Duration : {duration_1}\n {camera_view2} Duration : {duration_2}'
                        else:
                            discrepancies_found = None
                        return discrepancies_found

    def get_camera_fps_duration(self, camera_view_path):
        videos_camera_view = os.listdir(camera_view_path)
        videos_camera_view = [i for i in videos_camera_view if i.lower().endswith(self.video_extensions)]
        fps_values = set()
        duration_values = set()
        for idx, video_file in enumerate(videos_camera_view):
            video_file_path = os.path.join(camera_view_path, video_file)
            integrity_details = self.video_integrity(video_file_path)
            fps, duration = integrity_details['fps'], integrity_details['duration']
            fps_values.add(fps)
            duration_values.add(duration)
        if len(fps_values) == 1 and len(duration_values) == 1:
            return next(iter(fps_values)), next(
                iter(duration_values))  # return the common fps if all videos have the same fps

        else:
            logger.warning(
                f'Multiple fps or duration values found in camera view {os.path.basename(camera_view_path)}: FPS: {fps_values}, Durations: {duration_values}')
            return fps_values, duration_values

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
        video_duration = frame_count / fps if fps else 0
        logger(
            f"The FPS of the video {video_name_} is {fps} with width x height: {frame_width}x{frame_height} and frame count: {frame_count}")
        integrity_details = {'fps': fps, 'width': frame_width, 'height': frame_height, 'count': frame_count,
                             'duration': video_duration
                             }
        cap.release()
        return integrity_details


if __name__ == '__main__':
    proj_repo = '/home/qub-hri/Documents/PHEO Waiting Data'
    Integrity = CheckVidIntegrity(proj_repo, 1, 10)
    Integrity.check_integrity()
