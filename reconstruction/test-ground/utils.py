import numpy as np
import cv2
import h5py


def load_data(data_path, dataset_name):
    """
    Load data from an HDF5 file.

    Parameters:
    - data_path (str): Path to the HDF5 file.
    - dataset_name (str): Name of the dataset to load.

    Returns:
    - np.array: Data loaded from the HDF5 file.
    """
    with h5py.File(data_path, 'r') as file:
        if dataset_name in file:
            data = file[dataset_name][:]
        else:
            raise ValueError(f'Dataset {dataset_name} not found in the file.')
    return data


import numpy as np


def normalized_to_pixel_coordinates_batch(norm_coords_left, norm_coords_right, K_left, K_right):
    """
    Convert normalized 2D coordinates to pixel coordinates for multiple frames from both left and right cameras,
    ignoring the Z coordinate from MediaPipe since it does not represent absolute depth.

    Parameters:
        norm_coords_left (np.array): Normalized coordinates from the left camera, shape (F, N, 3).
        norm_coords_right (np.array): Normalized coordinates from the right camera, shape (F, N, 3).
        K_left (np.array): The 3x3 intrinsic matrix for the left camera.
        K_right (np.array): The 3x3 intrinsic matrix for the right camera.

    Returns:
        tuple: Two arrays containing pixel coordinates for each frame for the left and right cameras respectively.
    """
    # Select only the x, y coordinates and convert to homogeneous coordinates
    ones_left = np.ones((norm_coords_left.shape[0], norm_coords_left.shape[1], 1))
    ones_right = np.ones((norm_coords_right.shape[0], norm_coords_right.shape[1], 1))
    homog_coords_left = np.concatenate([norm_coords_left[:, :, :2], ones_left], axis=2)
    homog_coords_right = np.concatenate([norm_coords_right[:, :, :2], ones_right], axis=2)

    # Convert to pixel coordinates using the intrinsic matrices
    pixel_coords_left = np.einsum('fij,jk->fik', homog_coords_left, K_left)
    pixel_coords_right = np.einsum('fij,jk->fik', homog_coords_right, K_right)

    # Normalize to convert from homogeneous to 2D coordinates
    pixel_coords_left = pixel_coords_left[:, :, :2] / pixel_coords_left[:, :, 2][:, :, np.newaxis]
    pixel_coords_right = pixel_coords_right[:, :, :2] / pixel_coords_right[:, :, 2][:, :, np.newaxis]

    return pixel_coords_left, pixel_coords_right


def norm_to_px(norm_coords, K):
    """
    Convert normalized coordinates to pixel coordinates
    using the intrinsic matrix.

    Parameters:
    - norm_coords (tuple): Normalized coordinates (x, y).
    - K (np.array): Intrinsic Camera matrix of the camera as a 3x3 array

    Returns:
    - np.array: Pixel coordinates (x, y) as N x 2 array
    """
    # Append a row of ones to handle homogenous coordinates
    ones = np.ones((norm_coords.shape[0], 1))
    homogenous_coords = np.hstack((norm_coords, ones))

    pixel_coords = K @ homogenous_coords.T  # 3 x N matrix

    pixel_coords = pixel_coords[:2, :] / pixel_coords[2, :]  # convert from homogenous to 2D
    return pixel_coords.T


def calculate_baseline(T):
    """
    Calculate the baseline from the translation vector.

    Parameters:
    - T (np.array): Translation vector.

    Returns:
    - float: Baseline.
    """
    return np.linalg.norm(T)


def load_stereo_calib(stereo_filepath, keys):
    """
    Load calibration data from a npz file.

    Parameters:
    - stereo_filepath (str): Path to the npz file containing calibration data.

    Returns:
    - dict: A dictionary containing the calibration data.
    """
    stereo_data = np.load(stereo_filepath)
    datasets = {}
    for key in keys:
        if key in stereo_data:
            datasets[key] = stereo_data[key]
        else:
            print(f'Key {key} not found in the calibration data.')
    return datasets


def get_video_dimensions(video_path):
    """
    Get the dimensions of a video file.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    - tuple: Video dimensions (width, height).
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dimensions = {'width': width, 'height': height}
    return dimensions


if __name__ == '__main__':
    # Test the normalized_to_px_trad function

    # Load calibration data
    calibration_data_path = '/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz'
    keys = ['left_mtx', 'right_mtx']
    calibration_data = load_stereo_calib(calibration_data_path, keys)
    # Test the normalized_to_px function
    K = calibration_data['left_mtx']
    K_left, K_right = calibration_data['left_mtx'], calibration_data['right_mtx']
    # print(K)
    norm_coords = np.array([[0.547383, 0.25394]])
    print(norm_to_px(norm_coords, K))

    # Load H5 data
    data_to_load = 'face_landmarks'
    left_data = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BS.hdf5'
    right_data = '/home/iamshri/Documents/Dataset/p01/CAM_LR/landmarks/BIAH_BS.hdf5'

    norm_coord_left = load_data(left_data, data_to_load)
    norm_coord_right = load_data(right_data, data_to_load)
    # Convert normalized coordinates to pixel coordinates
    pixel_coords_left, pixel_coords_right = normalized_to_pixel_coordinates_batch(norm_coord_left, norm_coord_right,
                                                                                  K_left, K_right)
    print(pixel_coords_left.shape, pixel_coords_right.shape)
    print(pixel_coords_left[0], pixel_coords_right[0])
    print(pixel_coords_left)
