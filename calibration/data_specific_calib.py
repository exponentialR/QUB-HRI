import os

import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv
from logger_utils import setup_calibration_video_logger

DICTIONARY = [
    'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000', 'DICT_5X5_50',
    'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
    'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
    'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
    'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16h5', 'DICT_APRILTAG_16H5',
    'DICT_APRILTAG_25h9', 'DICT_APRILTAG_25H9', 'DICT_APRILTAG_36h10',
    'DICT_APRILTAG_36H10', 'DICT_APRILTAG_36h11', 'DICT_APRILTAG_36H11',
    'DICT_ARUCO_MIP_36h12', 'DICT_ARUCO_MIP_36H12'
]


def create_dir(direc):
    return os.makedirs(direc, exist_ok=True) if not os.path.exists(direc) else None


def sep_direc_files(input_path):
    path_parts = input_path.split(os.sep)
    sep_dir_files_path = os.path.join(path_parts[-2], path_parts[-1])
    return sep_dir_files_path


def log_exists(filename, row):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for existing_row in reader:
            if all(str(cell).strip() == str(value) for cell, value in zip(existing_row, row)):
                return True
    return False


def is_empty(filename):
    return os.path.getsize(filename) == 0


class CalibrateCorrect:
    def __init__(self, proj_repo, squaresX, squaresY, square_size, markerLength,
                 dictionary='DICT_4X4_100', frame_interval_calib=5, save_every_n_frames=5,
                 participant_id_start=1, participant_id_last=100):
        """
       Initialize the CalibrateCorrect class.

       Parameters:
       - proj_repo (str): The project repository path.
       - squaresX (int): The number of squares along the X-axis on the Charuco board.
       - squaresY (int): The number of squares along the Y-axis on the Charuco board.
       - square_size (float): The size of each square on the Charuco board.
       - markerLength (float): The size of the ArUco markers on the Charuco board.
       - dictionary (str): The type of ArUco dictionary to use.
       - frame_interval_calib (int): The interval between frames to use for calibration.
       - save_every_n_frames (int): The interval between frames to save.
       - participant_id_start (int): The starting participant ID.
       - participant_id_last (int): The last participant ID.
        """
        self.start_participant = os.path.join(proj_repo, f'p{participant_id_start:02d}')
        self.end_participant = os.path.join(proj_repo, f'p{participant_id_last:02d}')
        self.participant_list = [os.path.join(proj_repo, f'p{participant_id:02d}') for participant_id in
                                 range(participant_id_start, participant_id_last + 1)]

        self.save_calib_frames = save_every_n_frames
        self.save_path_prefix = 'calib_param'
        self.frame_interval_calib = frame_interval_calib
        self.square_size = square_size
        print(f'Type of Square size : {type(self.square_size)}')
        self.pattern_size = (squaresX, squaresY)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        print('Pattern Size', self.pattern_size)
        self.board = cv2.aruco.CharucoBoard(self.pattern_size, self.square_size / 1000, markerLength / 1000,
                                            self.aruco_dict)
        self.min_corners = 10  # 10
        self.calibration_params = None  # To store calibration parameters
        self.current_frame_number = 0
        self.video_name = None
        self.correct_et_orig_dict = {}
        self.cap_orig = None
        self.cap_corrected = None
        self.calibrate_correct_dict = {}
        self.logger = setup_calibration_video_logger()
        self.video_parent_dir = None

    def singleCalibMultiCorrect(self):
        """
        Perform single calibration and multiple corrections.

        This function goes through each participant directory and calibrates based on the 'CALIBRATION.MP4' video.
        It then performs corrections on the other videos in the same directory.
        """
        os.makedirs('csvlogs') if not os.path.exists('csvlogs') else None
        # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        with open(f'csvlogs/calibrated_videos.csv', mode='a', newline='') as c_file:
            calibrated_videos_writer = csv.writer(c_file)
            if is_empty('csvlogs/calibrated_videos.csv'):
                calibrated_videos_writer.writerow(["ParticipantID", "CameraViews", "CorrectedVideos"])

            # Create or open a log file to log directories with more than 10 videos
            with open(f'csvlogs/video_stats.csv', mode='a', newline='') as v_file:
                # Write headers to log files
                video_stats_writer = csv.writer(v_file)
                if is_empty('csvlogs/video_stats.csv'):
                    video_stats_writer.writerow(["ParticipantID", "CameraViews", "VideosCount"])

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
                            else:
                                self.logger.info(
                                    f'Calibration file {calib_file} already exists for participant {os.path.basename(participant_id)} camera view {camera_view}')
                                calib_file_path = calib_file
                            self.logger.debug(
                                f'Calibration done for video {os.path.basename(participant_id)} CAMERA {camera_view}. File saved at {calib_file_path}')

                            if calib_file_path is None:
                                # self.logger.critical(f'Calibration file {calib_file_path} does not exist')
                                self.logger.critical('Calibration Failed. Exiting... MOVING TO NEXT ')
                                continue

                            else:
                                self.logger.info('STARTING VIDEO CORRECTION')
                                videos_camera_view = os.listdir(current_camera)
                                videos_camera_view = [i for i in videos_camera_view if i.endswith('MP4')]
                                # Perform Correction
                                if len(videos_camera_view) <= 10:
                                    total_vids_len = len(videos_camera_view)
                                    corrected_video_count = 0
                                    for idx, video_file in enumerate(videos_camera_view):
                                        if video_file.endswith('MP4'):
                                            video_file_path = os.path.join(current_camera, video_file)
                                            self.logger.info(
                                                f'Correcting Videos {idx + 1} of {total_vids_len} of {camera_view} of {os.path.basename(participant_id)}')
                                            output_video_path = self.process_video(video_file_path, calib_file_path)
                                            self.logger.debug(f'Corrected VideoPath: {output_video_path}')
                                            print(f'Corrected VideoPath: {output_video_path}')
                                            corrected_video_count += 1
                                    data_stats = [os.path.basename(participant_id), os.path.basename(current_camera),
                                                  corrected_video_count]
                                    if not log_exists(f'csvlogs/calibrated_videos.csv', data_stats):
                                        calibrated_videos_writer.writerow(data_stats)

                                else:
                                    more_than_10 = [os.path.basename(participant_id), os.path.basename(current_camera),
                                                    len(os.listdir(current_camera))]
                                    if not log_exists(f'csvlogs/video_stats.csv', more_than_10):
                                        video_stats_writer.writerow(more_than_10)
                                    self.logger.critical(f'Length of {camera_view} Videos more than 10')

    def calibrate_single_video(self, single_video_calib):
        """
       Calibrate a single video.

       Parameters:
       - single_video_calib (str): The path to the video to be calibrated.

       Returns:
       - str: The path to the saved calibration parameters.
       """
        self.video_parent_dir = os.path.dirname(single_video_calib)
        cap = cv2.VideoCapture(single_video_calib)
        if not cap.isOpened():
            extra_info = {'frame_number'}
            self.logger.error(f'Could not open Video File. Exiting...', )
            return
        frame_count = 0
        gray = None
        all_charuco_corners = []
        all_charuco_ids = []
        video_name = os.path.basename(single_video_calib).split('.')[0]

        if self.video_parent_dir is not None:
            rejected_folder = os.path.join(self.video_parent_dir, 'RejectedCalibFrames')
            calib_frames = os.path.join(self.video_parent_dir, 'CalibrationFrames')
            create_dir(rejected_folder)
            create_dir(calib_frames)
        else:
            raise Exception
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
        with tqdm(total=frame_total // self.frame_interval_calib, desc='Processing frames', position=0,
                  leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info(f'Done Processing Video {single_video_calib}')
                    break
                if frame_count % self.frame_interval_calib == 0:
                    pbar.update(1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
                    if len(corners) > 0:
                        debug_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray,
                                                                                                self.board,
                                                                                                charucoIds=ids)

                        if not ret or len(charuco_corners) <= self.min_corners:
                            # self.logger.warning(f'Frame {frame_count}- Not enough corners for interpolation')
                            rejected_frame_filename = os.path.join(rejected_folder, f'{video_name}_{frame_count}.png')
                            cv2.imwrite(rejected_frame_filename, debug_frame)

                        else:
                            all_charuco_corners.append(charuco_corners)
                            all_charuco_ids.append(charuco_ids)
                            if self.save_calib_frames is not None and frame_count % self.save_calib_frames == 0:
                                frame_filename = os.path.join(calib_frames,
                                                              f'{video_name}_{frame_count}.png')
                                cv2.imwrite(frame_filename, debug_frame)
                    else:
                        continue
                        # self.logger.critical(f"Frame {frame_count} - Not enough corners for interpolation")

                frame_count += 1

            save_path = f"{self.video_parent_dir}/{self.save_path_prefix}_{os.path.basename(single_video_calib).split('.')[0]}.npz"
            if len(all_charuco_corners) > 0 and len(all_charuco_ids) > 0:
                valid_views = [i for i, corners in enumerate(all_charuco_corners) if len(corners) >= 4]
                all_charuco_corners = [all_charuco_corners[i] for i in valid_views]
                all_charuco_ids = [all_charuco_ids[i] for i in valid_views]

                self.logger.info('COMPUTING EXTRINSIC AND INTRINSIC PARAMETERS')
                ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids,
                                                                                self.board, gray.shape[::-1], None,
                                                                                None)
                np.savez(save_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

            else:
                self.logger.info('NOT ENOUGH CORNERS FOR CALIBRATION. EXITING...')

            cap.release()
            return save_path

    def process_video(self, video_file_path, calib_file_path):
        """
        Process and correct a video based on calibration parameters.

        Parameters:
        - video_file_path (str): The path to the video file to be processed.
        - calib_file_path (str): The path to the calibration parameters.

        Returns:
        - str: The path to the corrected video.
        """
        if not os.path.exists(calib_file_path):
            self.logger.warrning((f'CALIBRATION DATA FILE {calib_file_path} NOT FOUND!', 'ERROR'))
            return

        with np.load(calib_file_path) as data:
            mtx, dist = data['mtx'], data['dist']

        video_name = os.path.basename(video_file_path)
        video_name_ = os.path.basename(video_file_path).split('.')[0]

        cap = cv2.VideoCapture(video_file_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print(f'Original frame width x height: {frame_width} x {frame_height}')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"The FPS of the video is {fps}")

        # new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height))

        output_video_path = os.path.join(os.path.dirname(video_file_path), f'{video_name_}_CC.avi')
        if os.path.exists(output_video_path):
            self.logger.info(f'Video Already corrected Proceeding to next Video')
            return output_video_path
        else:
            # x, y, w, h = roi
            # print(f'Width x Height: {w} x {h}')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            self.logger.info(f"Correcting {video_name}")
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
            with tqdm(total=frame_total, desc=f'Correcting Video {video_name_}', position=0, leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    corrected = cv2.undistort(frame, mtx, dist, None, mtx)
                    out.write(corrected)
                    pbar.update(1)

                cap.release()
                out.release()
                try:
                    os.remove(video_file_path)
                    self.logger.info(f'Successfully deleted video file: {video_file_path}')
                except Exception as e:
                    self.logger.error(f'Failed to delete the original video file: {video_file_path}. Error: {e}')
                return output_video_path


if __name__ == '__main__':
    proj_repo = '/home/qub-hri/Documents/PHEO Waiting Data'
    squareX = 16
    squareY = 11
    square_size = 33
    markerLength = 26
    dictionary = 'DICT_4X4_100'
    # participant_id_last = 10
    calib = CalibrateCorrect(proj_repo=proj_repo, squaresX=squareX, squaresY=squareY, square_size=square_size,
                             markerLength=markerLength,
                             dictionary=dictionary, participant_id_last=15, participant_id_start=11)
    calib.singleCalibMultiCorrect()
