import cv2
import numpy as np
import glob


def calibrate_camera_from_video(video_paths, save_path):
    all_charuco_corners = []
    all_charuco_ids = []

    # Define Charuco board parameters
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard_create(7, 5, 0.04, 0.02, aruco_dict)

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract every 10th frame
            if frame_count % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

                if len(corners) > 0:
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

                    if ret:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)

            frame_count += 1

        cap.release()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                                    gray.shape[::-1], None, None)
    np.savez(save_path, mtx=mtx, dist=dist)


if __name__ == "__main__":
    video_paths = glob.glob('calibration_videos/*.mp4')
    save_path = 'calibration_parameters.npz'
    calibrate_camera_from_video(video_paths, save_path)
