import numpy as np
import h5py


def normalized_to_pixel_coordinates_batch(norm_coords, intrinsic_matrix):
    """
    Convert normalized 2D coordinates to pixel coordinates for multiple frames,
    ignoring the Z coordinate from MediaPipe since it does not represent absolute depth.

    Parameters:
        norm_coords (np.array): Normalized coordinates, shape (F, N, 3) where F is the number of frames,
                                N is the number of keypoints, and 3 are the x, y, z coordinates.
        intrinsic_matrix (np.array): The 3x3 intrinsic matrix for the camera.

    Returns:
        np.array: Pixel coordinates for each frame, shape (F, N, 2).
    """
    # Extract just the x and y coordinates, ignoring z
    norm_coords_xy = norm_coords[:, :, :2]

    # Convert to homogeneous coordinates by appending ones
    ones = np.ones((norm_coords.shape[0], norm_coords.shape[1], 1))

    homogeneous_coords = np.concatenate([norm_coords_xy, ones], axis=2)
    # Convert to pixel coordinates using the intrinsic matrix
    pixel_coords_homogeneous = np.einsum('fij,jk->fik', homogeneous_coords, intrinsic_matrix)
    # print(pixel_coords_homogeneous[0])
    # Normalize to convert from homogeneous to 2D coordinates
    pixel_coords = pixel_coords_homogeneous[:, :, :2] / pixel_coords_homogeneous[:, :, 2][:, :, None]
    print(pixel_coords[0])
    return pixel_coords


def scale_intrinsic_matrix(K, original_res, new_res):
    """
    Scale an intrinsic camera matrix from an original resolution to a new resolution.

    Args:
    K (np.array): Original 3x3 intrinsic matrix.
    original_res (tuple): The original resolution (width, height).
    new_res (tuple): The new resolution (width, height).

    Returns:
    np.array: Scaled intrinsic matrix.
    """
    # Extract original dimensions
    original_width, original_height = original_res
    new_width, new_height = new_res

    # Compute scale factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Create a scaling matrix
    scaling_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    # Scale the intrinsic matrix
    K_scaled = scaling_matrix @ K

    return K_scaled


def load_and_process_hdf5(hdf5_path_left, hdf5_path_right, K_left, K_right):
    """
    Load data from HDF5 files for left and right cameras and convert to pixel coordinates.

    Parameters:
        hdf5_path_left (str): Path to the HDF5 file for the left camera.
        hdf5_path_right (str): Path to the HDF5 file for the right camera.
        K_left (np.array): Intrinsic matrix for the left camera.
        K_right (np.array): Intrinsic matrix for the right camera.

    Returns:
        dict: Pixel coordinates for the left and right cameras.
    """
    with h5py.File(hdf5_path_left, 'r') as file_left, h5py.File(hdf5_path_right, 'r') as file_right:
        data_left = file_left['face_landmarks'][:]  # Adjust dataset name as needed
        data_right = file_right['face_landmarks'][:]  # Adjust dataset name as needed

    # Convert normalized coordinates to pixel coordinates
    pixel_coords_left = normalized_to_pixel_coordinates_batch(data_left, K_left)
    pixel_coords_right = normalized_to_pixel_coordinates_batch(data_right, K_right)

    return {'left': pixel_coords_left, 'right': pixel_coords_right}


if __name__ == '__main__':
    # Example usage
    stereo_data = np.load('/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz')
    left_intrinsic_matrix, right_intrinsic_matrix = stereo_data['left_mtx'], stereo_data['right_mtx']
    left_intrinsic_matrix = scale_intrinsic_matrix(left_intrinsic_matrix, (3840, 2160), (1728, 972))
    right_intrinsic_matrix = scale_intrinsic_matrix(right_intrinsic_matrix, (3840, 2160), (1728, 972))
    left_data = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/LL_BIAH_BS.hdf5'
    right_data = '/home/iamshri/Documents/Dataset/p01/CAM_LR/landmarks/LR_BIAH_BS.hdf5'
    pixel_coords = load_and_process_hdf5(left_data, right_data, left_intrinsic_matrix, right_intrinsic_matrix)
    print(pixel_coords['left'].shape, pixel_coords['right'].shape)
    # print(pixel_coords['left'][0], pixel_coords['right'][0])
