import cv2
import numpy as np
import glob
from cv2 import aruco
import os


def adjust_parameters_for_original_size(mtx_resized, scaling_factor):
    mtx_original = mtx_resized.copy()
    mtx_original[0, 0] *= scaling_factor  # Adjust fx
    mtx_original[1, 1] *= scaling_factor  # Adjust fy
    mtx_original[0, 2] *= scaling_factor  # Adjust cx
    mtx_original[1, 2] *= scaling_factor  # Adjust cy
    return mtx_original


def calibrate_camera_from_video(video_paths, save_path):
    all_charuco_corners = []
    all_charuco_ids = []
    output_folder = 'calibration_frames'
    scaling_factor = 2.0

    # Define Charuco board parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((11, 16), 0.033, 0.026, aruco_dict)

    for idx, video_path in enumerate(video_paths):
        print(f"Processing {video_path}...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Couldn't open video file {video_path}")
            continue
        frame_count = 0
        gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Extract every 10th frame
            if frame_count % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
                print(corners)

                if len(corners) > 0:
                    debug_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                    cv2.imshow('Debug qqqFrame - Detected Markers', debug_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

                    if ret:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        print(f'Detected {len(charuco_corners)} Charuco corners and {len(charuco_ids)} Charuco IDs. ')

                        # Save the frame used for calibration
                        frame_filename = os.path.join(output_folder, f'calibration_frame_{idx}_{frame_count}.png')
                        cv2.imwrite(frame_filename, frame)
                    else:
                        print(f'Charuco corners could not be interpolated.')
                else:
                    print('No markers detected')
                    # show the frame for debugging
                    cv2.imshow('Debug frame', frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            frame_count += 1

        cap.release()

    if gray is None:
        print('Error; No frames were processed. Exiting.')
        return
    if len(all_charuco_corners) == 0:
        print('Error: No  charuco corners were detected. Exiting')
        return

    # Perform camera calibration
    ret, mtx_resized, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                                    gray.shape[::-1], None, None)
    mtx_original = adjust_parameters_for_original_size(mtx_resized, scaling_factor)

    np.savez(save_path, mtx=mtx_original, dist=dist)


if __name__ == "__main__":
    video_paths = glob.glob('calibration_videos/*.mp4')
    print(video_paths)
    save_path = 'calibration_parameters.npz'
    calibrate_camera_from_video(video_paths, save_path)
