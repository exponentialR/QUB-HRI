import os
import json
import argparse

import cv2
import h5py
import numpy as np
import logging
from tqdm import tqdm
from ultralytics import YOLO

COLOURS = {
    'WARNING': '\033[93m',
    'INFO': '\033[94m',
    'DEBUG': '\033[92m',
    'CRITICAL': '\033[91m',
    'ERROR': '\033[91m',
    'ENDC': '\033[0m'
}



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


class AppendBoundingBoxesToHDF5:
    def __init__(self, aerial_vid_path, hdf5_path, trained_model_path):
        self.aerial_video_path = aerial_vid_path
        self.hdf5_path = hdf5_path
        self.model = YOLO(trained_model_path)

    def normalize_gaze(self, gaze_points, frame_width, frame_height):
        normalized_gaze = np.array(gaze_points, dtype=np.float32)
        normalized_gaze[:, 0] /= frame_width  # Normalize x coordinates
        normalized_gaze[:, 1] /= frame_height  # Normalize y coordinates
        return normalized_gaze

    def append_bbox_normgaze(self):
        # Open the video file
        aerial_cap = cv2.VideoCapture(self.aerial_video_path)
        video_frame_count = int(aerial_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(aerial_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(aerial_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Open the HDF5 file in append mode
        with h5py.File(self.hdf5_path, 'a') as hdf5_file:
            hdf5_frame_count = hdf5_file['aerial_gaze'].shape[0] if 'aerial_gaze' in hdf5_file else 0

            # Log if frame counts do not match
            if video_frame_count != hdf5_frame_count:
                logging.warning(
                    f"Frame count mismatch: Video frames ({video_frame_count}) do not match HDF5 frames ({hdf5_frame_count}).")

            # Ensure the bounding box dataset is created or append to it if already exists
            if 'bounding_boxes' not in hdf5_file.keys():
                bounding_boxes_dset = hdf5_file.create_dataset("bounding_boxes", (video_frame_count, 36, 4), dtype='f')
            else:
                bounding_boxes_dset = hdf5_file['bounding_boxes']

            # Normalize the existing gaze data
            gaze_data = hdf5_file['aerial_gaze'][:]
            normalized_gaze_data = self.normalize_gaze(gaze_data, frame_width, frame_height)

            # Update the gaze data in the HDF5 file
            if 'normalized_gaze' not in hdf5_file.keys():
                gaze_dset = hdf5_file.create_dataset("normalized_gaze", data=normalized_gaze_data)
            else:
                gaze_dset = hdf5_file['normalized_gaze']
                gaze_dset[:] = normalized_gaze_data

            # Process each frame in the video
            for frame_idx in tqdm(range(min(video_frame_count, hdf5_frame_count)), desc="Processing Aerial Video"):
                success, frame = aerial_cap.read()
                if not success:
                    break

                # Perform YOLO detection
                results = self.model(frame, verbose=False)

                # Initialize bounding box array with zeros
                boxes = np.zeros((36, 4), dtype=np.float32)  # Initialize with zeros for up to 36 boxes
                num_boxes = len(results[0].boxes)

                if num_boxes > 36:
                    logging.warning(
                        f"Frame {frame_idx} has more than 36 bounding boxes. Only the first 36 will be stored.")

                for idx, box in enumerate(results[0].boxes[:36]):  # Ensure only up to 36 boxes are processed
                    x, y, w, h = box.xywhn[0].tolist()
                    boxes[idx] = [x, y, w, h]

                # Log the bounding boxes for debugging
                logging.debug(f"Frame {frame_idx} bounding boxes: {boxes}")

                # Store the bounding boxes in the HDF5 dataset
                bounding_boxes_dset[frame_idx] = boxes

        # Release the video capture object
        aerial_cap.release()
        logging.info("Bounding boxes and normalized gaze data appended to HDF5 file.")


class SubtaskSideviewFeatures:
    def __init__(self, subtask_dir, output_dir, camera_pairs='CAM_AV', batchsize=100, landmark_processed_json=None):
        self.subtask_vid_dir = subtask_dir
        self.output_subtask_dir = output_dir
        self.camera_pairs = camera_pairs
        self.batchsize = batchsize
        self.landmark_processed_json = landmark_processed_json
        self.landmark_processed = load_json(self.landmark_processed_json) if self.landmark_processed_json else {}

    def save_landmark_status(self):
        """
        Save the processed landmark status to JSON.
        """
        json_file_path = os.path.join(self.landmark_processed_json)
        save_json(json_file_path, self.landmark_processed)

    def extract_landmarks(self, model_path):
        subtask = os.path.basename(self.subtask_vid_dir)
        print(f'{self.subtask_vid_dir} subtask_vid_dir')
        # Get all the video files in the subtask directory
        current_subtask_vids = [file for file in os.listdir(self.subtask_vid_dir) if
                                file.endswith('.mp4') and file.split('-')[1].startswith(self.camera_pairs)]
        # Create the output directory if it doesn't exist
        # os.makedirs(os.path.join(self.output_subtask_dir, subtask), exist_ok=True)

        subtask_count_landmark = 0
        for aerial_vid_file in tqdm(current_subtask_vids, desc='Processing subtask files'):

            if self.landmark_processed.get(aerial_vid_file):
                continue

            else:
                aerial_vid_path = os.path.join(self.subtask_vid_dir, aerial_vid_file)

                aerial_vid_file = aerial_vid_file.split('mp4')
                aerial_hdf5_output = os.path.join(self.output_subtask_dir, aerial_vid_file[0] + 'h5')
                print(f'{aerial_vid_path} aerial_vid_path')
                print(aerial_hdf5_output + '   aerial_hdf5_output')
                landmarks = AppendBoundingBoxesToHDF5(aerial_vid_path, aerial_hdf5_output, model_path)
                landmarks.append_bbox_normgaze()
                subtask_count_landmark += 1
                self.landmark_processed[f'{aerial_vid_file[0]}mp4'] = True
                self.save_landmark_status()
                if subtask_count_landmark % self.batchsize == 0:
                    self.save_landmark_status()
                    logger.info(f'Landmarks extracted for subtask {subtask}',
                                extra={'task_name': 'Landmark Extraction', 'detail': 'Subtask Processing'})
                    print(f'{subtask_count_landmark} files processed for subtask {subtask}')
                    break


def main(args):
    subtask_dir = args.subtask_vid_dir
    output_dir = args.output_subtask_dir
    landmark_processed_json = os.path.join(os.path.dirname(output_dir), F'{os.path.basename(output_dir)}_processed.json')

    subtask_sideview_features = SubtaskSideviewFeatures(subtask_dir, output_dir,
                                                        landmark_processed_json=landmark_processed_json)
    MODEL_PATH = "/home/samueladebayo/PycharmProjects/LegoDetect/runs/detect/train8/weights/Lego_YOLO.pt"
    subtask_sideview_features.extract_landmarks(model_path=MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files for facial landmark extraction.")
    parser.add_argument('--subtask_vid_dir', type=str, help='Path to the subtask directory.')
    parser.add_argument('--output_subtask_dir', type=str, help= 'Path to the output directory where results will be saved.')

    args = parser.parse_args()
    main(args)
