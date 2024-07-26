import numpy as np
import h5py
import cv2
import os
import json
from tqdm import tqdm
import argparse
import logging

# PROJ_DIR = '/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK/CAM_LL/BHO'
# OUTPUT_DIR = '/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK/TRIANGULATED'
# CALIB_PARENT_DIR = '/home/samueladebayo/Downloads/stereo_data'
# BATCHSIZE = 100
LANDMARK_PROCESSED_JSON = '/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK'

COLOURS = {
    'WARNING': '\033[93m',
    'INFO': '\033[94m',
    'DEBUG': '\033[92m',
    'CRITICAL': '\033[91m',
    'ERROR': '\033[91m',
    'ENDC': '\033[0m'
}


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


logger = setup_calibration_video_logger(
    "Subtask-Sideview-Features-Logger",
    format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %('
               'message)s',
    extra_attrs=['task_name', 'detail'],
    error_log_file='landmark_extraction_log.txt',
    levels_to_save={logging.DEBUG, logging.INFO},
    console_level=logging.INFO
)

# Set up logging
logging.basicConfig(filename='processing_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_json(json_file):
    """
    Load data from a JSON file. If the file doesn't exist, return an empty dictionary.

    Args:
    json_file (str): Path to the JSON file.

    Returns:
    dict: The data from the JSON file or an empty dictionary if the file doesn't exist.
    """
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        return {}


def save_json(json_file, data, merge=True):
    """
    Save data to a JSON file. If merging is True, it loads existing data and merges it.

    Args:
    json_file (str): Path to the JSON file.
    data (dict): Data to be saved.
    merge (bool): Whether to merge with existing data in the file.
    """
    current_data = {}
    if merge and os.path.exists(json_file):
        current_data = load_json(json_file)

    # Update current data with new data
    current_data.update(data)

    with open(json_file, 'w') as f:
        json.dump(current_data, f, indent=4)  # Using indent for pretty printing


def load_keypoints(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        face_landmarks = f['face_landmarks'][:]  # shape (923, 478, 2)
        left_hand_landmarks = f['left_hand_landmarks'][:]  # shape (923, 21, 2)
        pose_landmarks = f['pose_landmarks'][:]  # shape (923, 33, 2)
        right_hand_landmarks = f['right_hand_landmarks'][:]  # shape (923, 21, 2)
    return face_landmarks, left_hand_landmarks, pose_landmarks, right_hand_landmarks


class TriangulateViews:
    def __init__(self, stereo_path, left_path, right_path, output_file):
        self.stereo_calib = np.load(stereo_path)
        self.left_path = left_path
        self.right_path = right_path
        self.R = self.stereo_calib['R']
        self.T = self.stereo_calib['T']
        self.P1 = np.dot(self.stereo_calib['left_mtx'], np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.P2 = np.dot(self.stereo_calib['right_mtx'], np.hstack((self.R, self.T.reshape(3, 1))))
        self.face_landmarks_left, self.left_hand_landmarks_left, self.pose_landmarks_left, self.right_hand_landmarks_left = load_keypoints(
            left_path)
        self.face_landmarks_right, self.left_hand_landmarks_right, self.pose_landmarks_right, self.right_hand_landmarks_right = load_keypoints(
            right_path)
        self.output_file = output_file

    def triangulate_points(self, pts_left, pts_right):
        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, pts_left, pts_right)
        points_4d = points_4d_hom / points_4d_hom[3]
        return points_4d[:3].T  # Return only the 3D points

    def get_3d_points_for_frame(self, frame_number):
        pts_left_face = self.face_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_face = self.face_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_face = self.triangulate_points(pts_left_face, pts_right_face)

        pts_left_left_hand = self.left_hand_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_left_hand = self.left_hand_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_left_hand = self.triangulate_points(pts_left_left_hand, pts_right_left_hand)

        pts_left_right_hand = self.right_hand_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_right_hand = self.right_hand_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_right_hand = self.triangulate_points(pts_left_right_hand, pts_right_right_hand)

        pts_left_pose = self.pose_landmarks_left[frame_number].reshape(-1, 2).T
        pts_right_pose = self.pose_landmarks_right[frame_number].reshape(-1, 2).T
        points_3d_pose = self.triangulate_points(pts_left_pose, pts_right_pose)

        return points_3d_face, points_3d_left_hand, points_3d_right_hand, points_3d_pose

    def save_3d_points_hdf5(self):
        num_frames = self.face_landmarks_left.shape[0]
        with h5py.File(self.output_file, 'w') as f:
            face_points_3d = f.create_dataset('face_points_3d', (num_frames, 478, 3), dtype='f')
            left_hand_points_3d = f.create_dataset('left_hand_points_3d', (num_frames, 21, 3), dtype='f')
            right_hand_points_3d = f.create_dataset('right_hand_points_3d', (num_frames, 21, 3), dtype='f')
            pose_points_3d = f.create_dataset('pose_points_3d', (num_frames, 33, 3), dtype='f')

            for frame in range(num_frames):
                points_3d_face, points_3d_left_hand, points_3d_right_hand, points_3d_pose = self.get_3d_points_for_frame(
                    frame)
                face_points_3d[frame] = points_3d_face
                left_hand_points_3d[frame] = points_3d_left_hand
                right_hand_points_3d[frame] = points_3d_right_hand
                pose_points_3d[frame] = points_3d_pose

        print(f"3D points saved to {self.output_file}")


class SubtaskTriangulateViews:
    def __init__(self, proj_dir, output_dir, calib_parent_dir=None, batchsize=100, landmark_processed_json=None):
        self.proj_dir = proj_dir  # Path to the directory containing the subtask data (e.g. '/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK/CAM_LR/BHO'
        self.output_dir = output_dir
        self.calib_parent_dir = calib_parent_dir
        self.batchsize = batchsize
        self.landmark_processed_json = landmark_processed_json
        self.landmark_processed = load_json(self.landmark_processed_json) if self.landmark_processed_json else {}
        self.calib_parent_dir = calib_parent_dir

    def save_landmark_status(self):
        """
        Save the processed landmark status to JSON.
        """
        json_file_path = os.path.join(self.landmark_processed_json)
        save_json(json_file_path, self.landmark_processed)

    def extract_landmarks(self):
        subtask = os.path.basename(self.proj_dir)
        subtask_dir = self.proj_dir
        current_subtask_files = [file for file in os.listdir(subtask_dir) if
                                 file.endswith('.h5') and file.split('-')[1].startswith('CAM_LR')]
        os.makedirs(os.path.join(self.output_dir, subtask), exist_ok=True)

        subtask_count_landmark = 0
        for side_video_file in tqdm(current_subtask_files, desc='Processing subtask files'):
            if self.landmark_processed.get(side_video_file):
                continue
            else:
                left_hdf5 = os.path.join(subtask_dir, side_video_file)
                right_hdf5 = os.path.join(subtask_dir.replace('CAM_LR', 'CAM_LL'),
                                          side_video_file.replace('CAM_LR', 'CAM_LL'))
                participant_number = side_video_file.split('_')[0].split('-')[0]
                calib_file = os.path.join(self.calib_parent_dir, f'{participant_number}_LL_LR_stereo.npz')
                output_file = os.path.join(self.output_dir, subtask, side_video_file.replace('CAM_LR', 'LL_LR-3d'))

                if os.path.exists(output_file):
                    print(f'{output_file} already exists')
                    continue
                else:
                    print(f'Stereo Calib File: {calib_file}')
                    if os.path.exists(calib_file):
                        triangulate_views = TriangulateViews(calib_file, left_hdf5, right_hdf5,
                                                             os.path.join(self.output_dir, subtask, side_video_file))
                        triangulate_views.save_3d_points_hdf5()
                        subtask_count_landmark += 1

                        self.landmark_processed[side_video_file] = True
                        self.save_landmark_status()
                        if subtask_count_landmark % self.batchsize == 0:
                            self.save_landmark_status()
                            logger.info(f'Landmarks extracted for subtask {subtask}',
                                        extra={'task_name': 'Landmark Extraction', 'detail': 'Subtask Processing'})
                            print(f'{subtask_count_landmark} files processed for subtask {subtask}')
                            break


def main(args):
    subtask_dir = args.subtask_dir
    output_dir = args.output_dir
    calib_parent_dir = args.calib_dir
    landmark_processed_json = os.path.join(output_dir, F'{os.path.basename(subtask_dir)}_processed.json')

    subtask_sideview_features = SubtaskTriangulateViews(subtask_dir, output_dir, calib_parent_dir=calib_parent_dir,
                                                        landmark_processed_json=landmark_processed_json)
    subtask_sideview_features.extract_landmarks()


# Test the TriangulateViews class
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video files for facial landmark extraction.")
    parser.add_argument('--subtask_dir', type=str, help='Path to the subtask directory.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory where results will be saved.',
                        )
    parser.add_argument('--calib_dir', type=str, help='Path to the calibration parameters directory.',
                        )

    args = parser.parse_args()
    print(f'CURRENT SUBTASK DIR {args.subtask_dir}')
    main(args)
