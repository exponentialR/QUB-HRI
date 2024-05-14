import cv2
import numpy as np


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

                # Compute object points for the detected corners
                obj_points = np.array([object_points[i] for i in charuco_ids_left.flatten()], dtype=np.float32)
                all_object_points.append(obj_points)

    # Perform stereo calibration
    if len(all_object_points) > 0 and len(all_charuco_corners_left) > 0 and len(all_charuco_corners_right) > 0:
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
            all_object_points, all_charuco_corners_left, all_charuco_corners_right,
            left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs,
            gray_left.shape[::-1], criteria=termination_criteria, flags=flags)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                    cameraMatrix2, distCoeffs2,
                                                    gray_left.shape[::-1], R, T)

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
                'right_mtx': cameraMatrix2, 'right_dist': distCoeffs2,
                'R': R, 'T': T, 'E': E, 'F': F,
                'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q}
    else:
        print(f'Not enough corners for stereo calibration. Exiting...')
        return None
