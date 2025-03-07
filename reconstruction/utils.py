import os.path
import os
import numpy as np
import cv2
from tqdm import tqdm
import h5py
import logging
import os

COLOURS = {
    'WARNING': '\033[93m',
    'INFO': '\033[94m',
    'DEBUG': '\033[92m',
    'CRITICAL': '\033[91m',
    'ERROR': '\033[91m',
    'ENDC': '\033[0m'
}


def convert_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_list(item) for item in data]
    else:
        return data


def save_to_hdf5(data_list, hdf5_path):
    with h5py.File(hdf5_path, 'w') as f:
        for idx, data in tqdm(enumerate(data_list), total=len(data_list), desc=f'Saving {os.path.basename(hdf5_path)}'):
            f.create_dataset(f'frame_{idx}', data=np.string_(data))
    print(f'Saved {len(data_list)} frames to {hdf5_path}')


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


def create_dir(directory):
    """Utility function to create directory if it doesn't exist.

    Parameters:
        - directory: The directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def remove_files_in_folder(folder_path, file_ext=('png', 'jpg', 'jpeg')):
    """
   Remove all files in a specified folder path

   :param folder_path: Path to the folder from which files will be deleted
   :param file_ext: Tuple of file extensions to delete
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


def extract_landmarks(face_landmarks):
    flattened_landmarks = []
    for lm in face_landmarks.landmark:
        flattened_landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(flattened_landmarks)


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def convert_to_structured_array(face_landmarks):
    # Define the structured array data format
    dtypes = [(key, 'f8', (3,)) for key in face_landmarks.keys()]
    structured_array = np.zeros((1,), dtype=dtypes)

    # Fill the structured array with data
    for key, value in face_landmarks.items():
        structured_array[key] = value

    return structured_array


def normalize(vector):
    return vector / np.linalg.norm(vector)


def calculate_gaze_vector(left_iris_3d, right_iris_3d, forehead_3d, rot_vec):
    left_iris_3d = np.array(left_iris_3d)
    right_iris_3d = np.array(right_iris_3d)

    # Dynamically calculate the distance between the eyes
    eye_distance = np.linalg.norm(left_iris_3d - right_iris_3d)

    # Define eye centers based on the dynamic eye distance
    E_left = np.array([forehead_3d[0] - eye_distance / 2, forehead_3d[1], forehead_3d[2]])
    E_right = np.array([forehead_3d[0] + eye_distance / 2, forehead_3d[1], forehead_3d[2]])

    # Gaze vectors for each eye
    gaze_vector_left = normalize(left_iris_3d - E_left)
    gaze_vector_right = normalize(right_iris_3d - E_right)

    # Average gaze vector
    combined_gaze_vector = (gaze_vector_left + gaze_vector_right) / 2

    # Rotate gaze vector according to head pose
    rmat, _ = cv2.Rodrigues(rot_vec)
    world_gaze_vector = rmat.dot(combined_gaze_vector)

    return world_gaze_vector


def load_calibration_data(video_path):
    calibration_path = os.path.join(os.path.dirname(video_path), 'calib_param_CALIBRATION.npz')
    try:
        with np.load(calibration_path) as data:
            mtx, dist, rvecs, tvecs = data['mtx'], data['dist'], data['rvecs'], data['tvecs']
            return mtx, dist, rvecs, tvecs
    except IOError:
        print(f'Error Loading Claibration data from {calibration_path}')
        return None, None, None, None


class FlexibleLevelFilter(logging.Filter):
    def __init__(self, allowed_levels):
        super().__init__()
        self.allowed_levels = allowed_levels

    def filter(self, record):
        # Allow only specified levels
        return record.levelno in self.allowed_levels


class DynamicVideoFormatter(logging.Formatter):
    def __init__(self, format_str, extra_attrs):
        super().__init__(format_str)
        self.extra_attrs = extra_attrs

    def format(self, record):
        # Ensure extra attributes have default values
        for attr in self.extra_attrs:
            setattr(record, attr, getattr(record, attr, 'N/A'))
        log_message = super().format(record)
        return f"{COLOURS[record.levelname]}{log_message}{COLOURS['ENDC']}"
        # return f"{log_message}"  # Simplified formatting


def setup_calibration_video_logger(logger_name, format_str, extra_attrs, error_log_file, levels_to_save, console_level):
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(levels_to_save))  # Set logger level to the lowest level specified

    # Clear existing handlers
    logger.handlers.clear()

    # File handler setup
    file_handler = logging.FileHandler(error_log_file)
    file_handler.setLevel(min(levels_to_save))  # Set to the minimum level in levels_to_save
    file_handler.addFilter(FlexibleLevelFilter(levels_to_save))

    # Console handler setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)  # Control console level dynamically
    console_handler.addFilter(FlexibleLevelFilter(levels_to_save))  # Ensure consistent filtering with file

    formatter = DynamicVideoFormatter(format_str, extra_attrs)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
