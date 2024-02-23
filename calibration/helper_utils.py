import os.path
import os
import numpy as np
import cv2
from tqdm import tqdm


def create_dir(directory):
    """Utility function to create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def calibrate_single_video(single_video_calib, aruco_dict, board, frame_interval_calib, min_corners,
                                      save_calib_frames=None, max_calib_frames=50,
                                      save_path_prefix="CalibrationResult"):
    """
    Standalone function to calibrate a single video.

    Parameters:
    - single_video_calib (str): The path to the video to be calibrated.
    - aruco_dict: The dictionary of ArUco markers used for the detection.
    - board: The Charuco board that is used for calibration.
    - frame_interval_calib (int): The interval between frames to consider for calibration.
    - min_corners (int): Minimum number of corners required for considering a frame for calibration.
    - save_calib_frames (int, optional): Interval to save frames used for calibration. If None, frames are not saved.
    - max_calib_frames (int): Maximum number of frames to use for calibration.
    - save_path_prefix (str): Prefix for the saved calibration result file.

    Returns:
    - str: The path to the saved calibration parameters.
    """

    video_parent_dir = os.path.dirname(single_video_calib)
    cap = cv2.VideoCapture(single_video_calib)
    if not cap.isOpened():
        print('Could not open Video File. Exiting...')
        return

    frame_count = 0
    calib_frame_count = 0
    all_charuco_corners = []
    all_charuco_ids = []
    video_name = os.path.basename(single_video_calib).split('.')[0]

    rejected_folder = os.path.join(video_parent_dir, 'RejectedCalibFrames')
    calib_frames = os.path.join(video_parent_dir, 'CalibrationFrames')
    create_dir(rejected_folder)
    create_dir(calib_frames)

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=frame_total // frame_interval_calib, desc='Processing frames', position=0, leave=False) as pbar:
        while True:
            if calib_frame_count >= max_calib_frames:
                print(f'Maximum calibration frames ({max_calib_frames}) reached. Stopping...')
                break
            ret, frame = cap.read()
            if not ret:
                print(f'Done Processing Video {single_video_calib}')
                break
            if frame_count % frame_interval_calib == 0:
                pbar.update(1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
                if len(corners) > 0:
                    debug_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board,
                                                                                            charucoIds=ids)
                    if not ret or len(charuco_corners) <= min_corners:
                        rejected_frame_filename = os.path.join(rejected_folder, f'{video_name}_{frame_count}.png')
                        cv2.imwrite(rejected_frame_filename, debug_frame)
                    else:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        calib_frame_count += 1
                        if save_calib_frames is not None and frame_count % save_calib_frames == 0:
                            frame_filename = os.path.join(calib_frames, f'{video_name}_{frame_count}.png')
                            cv2.imwrite(frame_filename, debug_frame)
                frame_count += 1

    save_path = os.path.join(video_parent_dir, f"{save_path_prefix}_{video_name}.npz")
    if len(all_charuco_corners) > 0 and len(all_charuco_ids) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                                        gray.shape[::-1], None, None)
        if ret:
            np.savez(save_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print('Calibration parameters computed and saved.')
        else:
            print('Calibration failed.')
    else:
        print('Not enough corners for calibration. Exiting...')

    cap.release()
    return save_path
