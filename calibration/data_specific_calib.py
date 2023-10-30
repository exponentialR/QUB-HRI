import os

import cv2
import numpy as np

from utils.logger_utils import setup_calibration_video_logger


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


class CalibrateCorrect:
    def __init__(self, proj_repo, squaresX, squaresY, square_size, markerLength,
                 dictionary='DICT_4X4_250', frame_interval_calib=5, save_every_n_frames=5,
                 participant_id_start=1, participant_id_last=100):
        self.start_participant = os.path.join(proj_repo, f'p{participant_id_start:02d}')
        self.end_participant = os.path.join(proj_repo, f'p{participant_id_last:02d}')
        self.participant_list = [os.path.join(proj_repo, f'{participant_id:02d}') for participant_id in
                                 range(participant_id_start, participant_id_last+1)]

        self.save_calib_frames = save_every_n_frames
        self.save_path_prefix = 'calib_param'
        self.frame_interval_calib = frame_interval_calib
        self.square_size = square_size
        self.pattern_size = (squaresX, squaresY)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
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
        for participant_id in sorted(self.participant_list):
            for camera_view in os.listdir(participant_id):  # CAM_AV, CAM_LL, CAM_UL,
                current_camera = os.path.join(participant_id, camera_view)
                self.logger.info(f'CURRENT CAMERA DIRECTORY- {current_camera}')

                # Perform Calibration
                calib_video_file = os.path.join(current_camera, 'CALIBRATION.MP4')
                calib_file_path = self.calibrate_single_video(calib_video_file)
                self.logger.debug(
                    f'Calibration done for video {os.path.basename(calib_video_file)}. File saved at {calib_file_path}')
                if calib_file_path is None:
                    self.logger.critical(f'Calibration file {calib_file_path} does not exist')
                    self.logger.critical('Calibration Failed. Exiting... MOVING TO NEXT ')
                    continue
                else:
                    self.logger.info('STARTING VIDEO CORRECTION', 'DEBUG')
                    videos_camera_view = os.listdir(current_camera)

                    # Perform Correction
                    if len(videos_camera_view) <= 10:
                        total_vids_len = len(videos_camera_view)
                        for idx, video_file in videos_camera_view:
                            if video_file.endswith('MP4'):
                                video_file_path = os.path.join(current_camera, video_file)
                                self.logger.info(f'Correcting Videos {idx + 1} of {total_vids_len}')
                                output_video_path = self.process_video(video_file_path, calib_file_path)
                                self.correct_et_orig_dict[video_file] = output_video_path
                                save_path_ = sep_direc_files(calib_file_path)
                                output_vid_path_ = sep_direc_files(output_video_path)
                                self.calibrate_correct_dict[video_file] = [save_path_, output_vid_path_]
        self.correct_et_orig_dict.clear()

    def calibrate_single_video(self, single_video_calib):
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
        else:
            raise Exception

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.info(f'Done Processing Video {single_video_calib}', 'INFO')
                break
            if frame_count % self.frame_interval_calib == 0:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
                if len(corners) > 0:
                    debug_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray,
                                                                                            self.board,
                                                                                            charucoIds=ids)

                    if not ret or len(charuco_corners) <= self.min_corners:
                        self.logger.warning(f'Frame {frame_count}- Not enough corners for interpolation')
                        rejected_frame_filename = os.path.join(rejected_folder, f'{video_name}_{frame_count}.png')
                        cv2.imwrite(rejected_frame_filename, debug_frame)

                    else:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        if self.save_calib_frames is not None and frame_count % self.save_calib_frames == 0:
                            frame_filename = os.path.join(calib_frames, video_name,
                                                          f'{video_name}_{frame_count}.png')
                            cv2.imwrite(frame_filename, debug_frame)
                else:
                    self.logger.critical(f"Frame {frame_count} - Not enough corners for interpolation")

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
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_video_path = f'{video_file_path}/{video_name_}.MP4'
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                              (frame_width, frame_height))
        self.logger.info(f"Correcting {video_name}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            corrected = cv2.undistort(frame, mtx, dist, None, mtx)
            out.write(corrected)

        cap.release()
        out.release()
        return output_video_path


if __name__ == '__main__':
    proj_repo = ''
    squareX = 16
    squareY = 11
    square_size = 33
    markerLength = 26,
    dictionary = 'DICT_4X4_250'
    participant_id_last = 5
    calib = CalibrateCorrect(proj_repo=proj_repo, squaresX=squareX, squaresY=squareY, square_size=square_size,
                             markerLength=markerLength,
                             dictionary=dictionary, participant_id_last=participant_id_last)
