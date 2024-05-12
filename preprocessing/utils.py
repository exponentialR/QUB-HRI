import os
import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
import logging

# ANSI escape codes for coloured terminal text
COLOURS = {
    'WARNING': '\033[93m',
    'INFO': '\033[94m',
    'DEBUG': '\033[92m',
    'CRITICAL': '\033[91m',
    'ERROR': '\033[91m',
    'ENDC': '\033[0m'
}


def reduce_resolution(input_video_path, output_video_path, scale_percent):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # calcuate the the new dimensions
    width = int(frame_width * scale_percent / 100)
    height = int(frame_height * scale_percent / 100)
    new_dim = (width, height)
    print(f"New Dimensions: {new_dim}")

    if not os.path.exists(output_video_path):
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, new_dim)

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
              desc=f"Reducing Resolution for {os.path.basename(input_video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
            out.write(resized_frame)
            pbar.update(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def display_video(video_path, window_name):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set the window name as the video name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Reached the end of the video or an error occurred. Exiting.")
            break

        # Display the resulting frame
        cv2.imshow(window_name, frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close any opened windows
    cap.release()
    cv2.destroyAllWindows()


class DebugWarningErrorFilter(logging.Filter):
    def filter(self, record):
        # Allow only debug, warning, and error levels (exclude info and critical)
        return record.levelno in (logging.DEBUG, logging.WARNING, logging.ERROR)


class DynamicVideoFormatter(logging.Formatter):
    def __init__(self, format_str, extra_attrs):
        super().__init__(format_str)
        self.extra_attrs = extra_attrs

    def format(self, record):
        # Set default values for extra attributes
        for attr in self.extra_attrs:
            setattr(record, attr, getattr(record, attr, 'N/A'))

        log_message = super().format(record)
        return f"{COLOURS[record.levelname]}{log_message}{COLOURS['ENDC']}"


def extract_name(full_path):
    components = full_path.split('/')
    desired_path = ''
    for component in components:
        if component.startswith('p') or component.startswith('CAM_') or component.endswith('.mp4'):
            desired_path = os.path.join(desired_path, component)
    return desired_path


def setup_calibration_video_logger(logger_name, format_str, extra_attrs, error_log_file):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Logger level

    # Clear existing handlers
    logger.handlers.clear()

    # File handler for specific task
    file_handler = logging.FileHandler(error_log_file)
    file_handler.setLevel(logging.DEBUG)  # Capture debug and above
    file_handler.addFilter(DebugWarningErrorFilter())  # Custom filter for file handler

    # Console handler for output to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set higher than DEBUG to exclude debug messages

    formatter = DynamicVideoFormatter(format_str, extra_attrs)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Testing
if __name__ == "__main__":
    input_video_path = 'data/input/BIAH_RB.mp4'
    output_video_path = 'data/output/output.mp4'
    reduce_resolution(input_video_path, output_video_path, 45)
    display_video(output_video_path, 'Reduced Resolution Video')
