import numpy as np
import cv2
from utils import norm_to_px, calculate_baseline, load_stereo_calib, normalized_to_pixel_coordinates_batch


def calculate_3d_coordinates(px_coord_left, px_coord_right, K_left, baseline):
    """
    Calculate the disparity, depth, and convert 2D pixel coordinates to 3D coordinates using intrinsic matrix.

    Parameters:
        px_coord_left (np.array): Pixel coordinates from the left camera, shape (N, 2).
        px_coord_right (np.array): Pixel coordinates from the right camera, shape (N, 2).
        K_left (np.array): The 3x3 intrinsic matrix of the left camera.
        baseline (float): The distance between the two camera centers.

    Returns:
        np.array: 3D coordinates of landmarks, shape (N, 3).
    """
    # Extract focal length and optical centers from the intrinsic matrix
    f = K_left[0, 0]  # Assuming fx = fy for simplicity
    c_x, c_y = K_left[0, 2], K_left[1, 2]

    # Calculate disparity as the difference in the x-coordinates
    disparity = px_coord_left[:, 0] - px_coord_right[:, 0]

    # Handle zero disparity to avoid division by zero
    disparity[disparity == 0] = 1e-5  # A small value to prevent division by zero

    # Calculate depth using the formula: Z = (f * B) / d
    depth = (f * baseline) / disparity

    # Calculate real-world X, Y, Z coordinates
    X = (px_coord_left[:, 0] - c_x) * depth / f
    Y = (px_coord_left[:, 1] - c_y) * depth / f
    Z = depth  # depth is the real-world Z coordinate

    # Combine into a single N x 3 array
    three_d_coords = np.vstack((X, Y, Z)).T

    return three_d_coords


if __name__ == '__main__':
    # Load stereo calibration data
    stereo_filepath = '/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz'
    stereo_data = load_stereo_calib(stereo_filepath, ['left_mtx', 'right_mtx', 'T'])
    # load the data
    K_left, K_right = stereo_data['left_mtx'], stereo_data['right_mtx']
    baseline = calculate_baseline(stereo_data['T'])
    # load normalized coordinates
    data_path_left, data_path_right = 'dataleft.npy', 'dataright.npy'
    norm_coords_left, norm_coords_right = np.load(data_path_left), np.load(data_path_right)
    print(norm_coords_left.shape, norm_coords_right.shape)
    # convert normalized coordinates to pixel coordinates

    # px_coords_left = norm_to_px(norm_coords_left, K_left)
    # px_coords_right = norm_to_px(norm_coords_right, K_right)
    # calculate 3D coordinates
    # three_d_coords = calculate_3d_coordinates(px_coords_left, px_coords_right, K_left, baseline)
    # print(three_d_coords)




