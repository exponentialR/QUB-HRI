import cv2
import numpy as np
import glob
from cv2 import aruco
import os
from argparse import ArgumentParser

# Argument parser for command-line options
parser = ArgumentParser(description='Calibrate camera using a video with Charuco board.')
parser.add_argument('--squaresX', type=int, default=7, help='Number of chessboard squares along the X-axis')
parser.add_argument('--squaresY', type=int, default=5, help='Number of chessboard squares along the Y-axis')
parser.add_argument('--squareLength', type=float, default=0.04,
                    help='Length of a square in the Charuco board (in meters)')
parser.add_argument('--markerLength', type=float, default=0.02,
                    help='Length of a marker in the Charuco board (in meters)')
parser.add_argument('--dictionary', type=str, default='DICT_4X4_50', choices=[
    'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
    'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
    'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
    'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
    'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16h5', 'DICT_APRILTAG_16H5',
    'DICT_APRILTAG_25h9', 'DICT_APRILTAG_25H9', 'DICT_APRILTAG_36h10',
    'DICT_APRILTAG_36H10', 'DICT_APRILTAG_36h11', 'DICT_APRILTAG_36H11',
    'DICT_ARUCO_MIP_36h12', 'DICT_ARUCO_MIP_36H12'
], help='Dictionary used for generating the Charuco board')
args = parser.parse_args()


def adjust_parameters_for_original_size(mtx_resized, scaling_factor):
    mtx_original = mtx_resized.copy()
    mtx_original[0, 0] *= scaling_factor  # Adjust fx
    mtx_original[1, 1] *= scaling_factor  # Adjust fy
    mtx_original[0, 2] *= scaling_factor  # Adjust cx
    mtx_original[1, 2] *= scaling_factor  # Adjust cy
    return mtx_original


def calibrate_camera_from_video(video_paths, save_path, squaresX=args.squaresX, squaresY=args.squaresY,
                                squareLength=args.squareLength, markerLength=args.markerLength,
                                dictionary=args.dictionary):

    all_charuco_corners = []
    all_charuco_ids = []
    output_folder = 'calibration_frames'
    os.makedirs(output_folder) if not os.path.exists(output_folder) else None

    # Define Charuco board parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength/1000, markerLength/1000, aruco_dict)
    min_corners = 10

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

            if len(corners) > 0:

                debug_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                cv2.waitKey(10)
                cv2.destroyAllWindows()

                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board,
                                                                                        charucoIds=ids)

                if not ret or len(charuco_corners) <= min_corners:
                    print(f'Not enough corners for interpolation')
                else:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    print(f'Detected {len(charuco_corners)} Charuco corners and {len(charuco_ids)} Charuco IDs. ')

                    # Save the frame used for calibration
                    frame_filename = os.path.join(output_folder, f'calibration_frame_{idx}_{frame_count}.png')
                    cv2.imwrite(frame_filename, debug_frame)
            else:
                print('No markers detected')
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
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                                    gray.shape[::-1], None, None)

    np.savez(save_path, mtx=mtx, dist=dist)


if __name__ == "__main__":
    video_paths = glob.glob('calibration_videos/*.MP4')
    print(video_paths)
    save_path = 'calibration_parameters.npz'
    calibrate_camera_from_video(video_paths, save_path)
