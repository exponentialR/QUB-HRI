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
                           max_calib_frames=50, save_path_prefix="CalibrationResult"):
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
    calib_frames_folder = os.path.join(video_parent_dir, 'CalibrationFrames')
    create_dir(rejected_folder)
    create_dir(calib_frames_folder)

    expected_updates = max_calib_frames

    with tqdm(total=expected_updates, desc='Processing frames', position=0, leave=False) as pbar:
        while calib_frame_count < max_calib_frames:
            ret, frame = cap.read()
            if not ret:
                print(f'Done Processing Video {single_video_calib}')
                break

            if frame_count % frame_interval_calib == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

                if len(corners) > 0:
                    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board, charucoIds=ids)
                    if charuco_corners is not None and len(charuco_corners) > min_corners:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        calib_frame_count += 1
                        pbar.update(1)

                        frame_filename = os.path.join(calib_frames_folder, f'{video_name}_{frame_count}.png')
                        cv2.imwrite(frame_filename, frame)
                    else:
                        rejected_frame_filename = os.path.join(rejected_folder, f'{video_name}_{frame_count}.png')
                        cv2.imwrite(rejected_frame_filename, frame)
            frame_count += 1

    save_path = os.path.join(video_parent_dir, "calibration.npz")
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
    cv2.destroyAllWindows()  # Ensure all created windows are destroyed
    return save_path


if __name__ == '__main__':
    calibrate_single_video('/home/iamshri/Documents/PHEO-Data/Unprocessed/p01/CAM_LL/CALIBRATION.MP4',
                           cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
                           cv2.aruco.CharucoBoard((16, 11), 33 / 1000, 26 / 1000,
                                                  cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
                           5, 10, 5, 100, 'CalibrationResult')
