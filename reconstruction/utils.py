import os.path
import os
import numpy as np
import cv2
from tqdm import tqdm


def create_dir(directory):
    """Utility function to create directory if it doesn't exist.

    Parameters:
        - directory: The directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def remove_files_in_folder(folder_path, file_ext):
   """
   Remove all files in a specified folder path

   :param folder_path: Path to the folder from which files will be deleted
   :return:
   """
   files = os.listdir(folder_path)
   for file in files:
       file_path = os.path.join(folder_path, file)
       try:
            if os.path.isfile(file_path) and file.endswith((file_ext)):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"Skipped deleting: {file_path} as it's not a file.")
        except Exception as e:
            print(f"Failed to delete: {file_path} due to {e}")


def extrinsic_calibration(left_calibration_data, right_calibration_data, stereo_image_pairs, board,
                          termination_criteria, flags):
    """
    Perform extrinsic or stereo calibration using previously obtained individual camera calibration data.
    This process aligns the coordinate systems of two cameras, allowing for depth estimation and 3D reconstruction.

    Parameters:
    - left_calibration_data (str): Path to the npz file containing calibration data for the left camera.
    - right_calibration_data (str): Path to the npz file containing calibration data for the right camera.
    - stereo_image_pairs (list of tuples): Pairs of image paths for left and right images used in calibration.
    - board (cv2.aruco.CharucoBoard): The Charuco board object used for calibration.
    - termination_criteria (cv2.TermCriteria): Criteria for the termination of the stereo calibration algorithm.
    - flags (int): Flags used by the stereoCalibrate function to control the calibration process.

    Returns:
    - dict: A dictionary containing stereo calibration parameters if successful, None otherwise.
    """

    # Load individual camera calibration data
    left_calib = np.load(left_calibration_data)
    right_calib = np.load(right_calibration_data)
    left_camera_matrix, left_dist_coeffs = left_calib['mtx'], left_calib['dist']
    right_camera_matrix, right_dist_coeffs = right_calib['mtx'], right_calib['dist']

    # Prepare object points
    # Generate object points for the Charuco board just once, as they are the same for every image
    object_points = board.getChessboardCorners()
    all_charuco_corners_left = []
    all_charuco_corners_right = []
    all_charuco_ids = []
    all_object_points = []

    for left_img_path, right_img_path in stereo_image_pairs:
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        corners_left, ids_left, _ = cv2.aruco.detectMarkers(gray_left, board.getDictionary())
        corners_right, ids_right, _ = cv2.aruco.detectMarkers(gray_right, board.getDictionary())

        if len(corners_left) > 0 and len(ids_left) > 0 and len(corners_right) > 0 and len(ids_right) > 0:
            _, charuco_corners_left, charuco_ids_left = cv2.aruco.interpolateCornersCharuco(corners_left, ids_left,
                                                                                            gray_left, board)
            _, charuco_corners_right, charuco_ids_right = cv2.aruco.interpolateCornersCharuco(corners_right, ids_right,
                                                                                              gray_right, board)

            if charuco_corners_left is not None and charuco_corners_right is not None and np.array_equal(
                    charuco_ids_left, charuco_ids_right):
                all_charuco_corners_left.append(charuco_corners_left)
                all_charuco_corners_right.append(charuco_corners_right)
                all_charuco_ids.append(charuco_ids_left)

                # Compute object points for the detected corners
                obj_points = np.array([object_points[i] for i in charuco_ids_left.flatten()], dtype=np.float32)
                all_object_points.append(obj_points)

    # Perform stereo calibration
    if len(all_object_points) > 0 and len(all_charuco_corners_left) > 0 and len(all_charuco_corners_right) > 0:
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
            all_object_points, all_charuco_corners_left, all_charuco_corners_right,
            left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
            gray_left.shape[::-1], criteria=termination_criteria, flags=flags)

        print(f'Stereo Calibration Successful| Reprojection Error: {retval}')
        print(f'Camera Matrix 1: {cameraMatrix1}')
        print(f'Distortion Coefficients 1: {distCoeffs1}')
        print(f'Camera Matrix 2: {cameraMatrix2}')
        print(f'Distortion Coefficients 2: {distCoeffs2}')
        print(f'R: {R}')
        print(f'T: {T}')
        print(f'E: {E}')
        print(f'F: {F}')

        return {'retval': retval, 'left_mtx': cameraMatrix1, 'left_dist': distCoeffs1,
                'right_mtx': cameraMatrix2, 'right_dist': distCoeffs2, 'R': R, 'T': T, 'E': E, 'F': F}
    else:
        print(f'Not enough corners for stereo calibration. Exiting...')
        return None


def extract_synchronized_frames(left_video_path, right_video_path, aruco_dict, board, frame_interval=2, min_corners=5,
                                max_frames=500):
    save_left_path = os.path.join(os.path.dirname(left_video_path), 'StereoCalibFrames')
    save_right_path = os.path.join(os.path.dirname(right_video_path), 'StereoCalibFrames')
    create_dir(save_right_path)
    create_dir(save_left_path)
    if len(os.listdir(save_left_path)) >= max_frames or len(os.listdir(save_right_path)) >= max_frames:
        print(f'{max_frames} Frames have already been extracted from {os.path.basename(left_video_path)} and '
              f'{os.path.basename(right_video_path)}. Skipping...')
        return save_left_path, save_right_path

    if len(os.listdir(save_left_path)) < max_frames or len(os.listdir(save_right_path)) > max_frames:
        remove_files_in_folder(save_left_path)
        remove_files_in_folder(save_right_path)

    cap_left = cv2.VideoCapture(left_video_path)
    cap_right = cv2.VideoCapture(right_video_path)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print(f'Could not open one or both video files')
        return

    left_frame_count = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frame_count = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_difference = abs(left_frame_count - right_frame_count)
    print(f'There are {frame_count_difference} frames difference between the two videos')
    print(
        f'attempting to synchronize videos {os.path.basename(left_video_path)} and {os.path.basename(right_video_path)}')

    if left_frame_count > right_frame_count:
        for _ in range(frame_count_difference):
            cap_left.read()  # Skip initial frames in left video
        print(f'Skip frame {frame_count_difference} in left video')
    elif right_frame_count > left_frame_count:
        for _ in range(frame_count_difference):
            cap_right.read()  # Skip initial frames in right video
        print(f'Skip frame {frame_count_difference} in right video')

    frame_count = 0
    saved_frame_count = 0

    left_fps = cap_left.get(cv2.CAP_PROP_FPS)
    right_fps = cap_right.get(cv2.CAP_PROP_FPS)
    common_fps = min(left_fps, right_fps)

    while saved_frame_count < max_frames:
        left_time = cap_left.get(cv2.CAP_PROP_POS_MSEC)
        right_time = cap_right.get(cv2.CAP_PROP_POS_MSEC)
        time_difference = abs(left_time - right_time)

        if time_difference > (1000 / common_fps):
            if left_time < right_time:
                cap_left.read()
            else:
                cap_right.read()
            continue

        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print(
                f'Done processing videos {os.path.basename(left_video_path)} and {os.path.basename(right_video_path)}')
            break

        if frame_count % frame_interval == 0:
            visible_left = detect_charuco_board(frame_left, aruco_dict, board, min_corners)
            visible_right = detect_charuco_board(frame_right, aruco_dict, board, min_corners)

            if visible_left and visible_right:
                cv2.imwrite(os.path.join(save_left_path, f"frame_{frame_count:04d}.png"), frame_left)
                cv2.imwrite(os.path.join(save_right_path, f"frame_{frame_count:04d}.png"), frame_right)
                saved_frame_count += 1
                print(f'Saved frame pair {saved_frame_count}')
        frame_count += 1
    if saved_frame_count < max_frames:
        print(f'Could not extract enough frames for stereo calibration. Extracted {saved_frame_count} frames')
    cap_left.release()
    cap_right.release()
    return save_left_path, save_right_path


def detect_charuco_board(frame, aruco_dict, board, min_corners):
    """
    Detects a Charuco board in the given frame
    Returns True if the board is detected with at least min_corners detected: otherwise Returns False
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_corners is not None and len(charuco_corners) >= min_corners:
            return True
    return False
