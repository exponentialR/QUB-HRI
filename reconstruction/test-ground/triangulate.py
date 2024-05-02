import numpy as np
import h5py


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


def convert_landmarks_adjusted(hdf5_path, K_scaled):
    """
    Convert normalized landmark data in an HDF5 file to pixel coordinates for varying data shapes.
    """
    with h5py.File(hdf5_path, 'r') as file:
        results = {}
        for landmark_type in file.keys():
            data = file[landmark_type][:]

            if landmark_type == 'hand_landmarks':
                # Handle two hands with multiple landmarks
                F, H, N, _ = data.shape  # Frames, Hands, Landmarks, Coordinates
                pixel_coords = np.zeros((F, H, N, 2))  # Prepare array for pixel coords
                for f in range(F):
                    for h in range(H):
                        frame_coords = data[f, h, :, :2]  # x, y coordinates
                        pixel_coords[f, h] = convert_to_pixels(frame_coords, K_scaled)
            else:
                # General case for face and pose
                F, N, _ = data.shape  # Frames, Landmarks, Coordinates
                pixel_coords = np.zeros((F, N, 2))
                for f in range(F):
                    frame_coords = data[f, :, :2]  # x, y coordinates
                    pixel_coords[f] = convert_to_pixels(frame_coords, K_scaled)

            results[landmark_type] = pixel_coords

    return results


def convert_to_pixels(frame_coords, K):
    """
    Helper function to convert normalized coordinates to pixel coordinates using intrinsic matrix.
    """
    ones = np.ones((frame_coords.shape[0], 1))
    homogeneous_norm_coords = np.hstack((frame_coords, ones))
    transformed_coords = K @ homogeneous_norm_coords.T
    return (transformed_coords[:2, :] / transformed_coords[2, :]).T


def check_hdf5_structure(hdf5_path):
    """
    Print the structure and shapes of datasets within an HDF5 file.

    Args:
    hdf5_path (str): Path to the HDF5 file.
    """
    with h5py.File(hdf5_path, 'r') as file:
        for key in file.keys():
            data = file[key]
            print(f"Dataset '{key}' shape: {data.shape}")
            # Optionally print more details or the data itself
            # print(data[:])


# Example usage
hdf5_path = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/LL_BIAH_BS.hdf5'
check_hdf5_structure(hdf5_path)
K = np.load('/home/iamshri/Documents/Dataset/p01/LL_LR_stereo.npz')['left_mtx']
# K_scaled = scale_intrinsic_matrix(K, (3849, 2160), (1728, 972))
landmark_pixels = convert_landmarks_adjusted(hdf5_path, K)
# landmark_pixels = convert_landmarks(hdf5_path, K_scaled)
print("Processed Landmark Pixel Coordinates:", landmark_pixels)
