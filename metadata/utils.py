import logging
import os
# ANSI escape codes for coloured terminal text
COLOURS = {
    'WARNING': '\033[93m',
    'INFO': '\033[94m',
    'DEBUG': '\033[92m',
    'CRITICAL': '\033[91m',
    'ERROR': '\033[91m',
    'ENDC': '\033[0m'
}


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


def process_cam_directory(cam_dir, task_metadata):
    calib_preext = ('intrinsic', 'calib')
    video_ext = ('AVI', 'MP4', 'mp4', 'avi')
    calibration_vid = [f for f in os.listdir(cam_dir) if
                       f.endswith(video_ext) and f.lower().startswith('calib')]
    intrinsic_calib_file = [f for f in os.listdir(cam_dir) if f.endswith('.npz') and f.startswith((calib_preext))]
    file_rename_map = {}
    file_count = 0
    if calibration_vid:
        calibration_vid = calibration_vid[0]
        calibration_vid_path = os.path.join(cam_dir, calibration_vid)
        new_calibration_vid_path = os.path.join(cam_dir, 'calibration.mp4')
        os.rename(calibration_vid_path, new_calibration_vid_path)
        print(f'{calibration_vid_path} renamed to {new_calibration_vid_path}')
        file_rename_map[calibration_vid] = 'calibration.mp4'
        file_count += 1

    if intrinsic_calib_file:
        intrinsic_calib_file = intrinsic_calib_file[0]
        intrinsic_calib_file_path = os.path.join(cam_dir, intrinsic_calib_file)
        new_intrinsic_calib_file_path = os.path.join(cam_dir, 'intrinsic.npz')
        os.rename(intrinsic_calib_file_path, new_intrinsic_calib_file_path)
        print(f'{intrinsic_calib_file_path} renamed to {new_intrinsic_calib_file_path}')
        file_rename_map[intrinsic_calib_file] = 'intrinsic.npz'
        file_count += 1
    try:
        video_file_list = [f for f in os.listdir(cam_dir) if
                           f.endswith((video_ext)) and f.startswith('GX') and not f.lower().startswith((calib_preext))]
        video_file_list.sort()
        new_names = [os.path.join(cam_dir, f'{i}.mp4') for i in task_metadata[1:]]
        video_paths = [os.path.join(cam_dir, f) for f in video_file_list]

        for old_name, new_name in zip(video_paths, new_names):
            os.rename(old_name, new_name)
            print(f'{old_name} renamed to {new_name}')
            file_rename_map[os.path.basename(old_name)] = os.path.basename(new_name)
            file_count += 1
        return file_rename_map, file_count

    except Exception as e:
        print(f'Error processing {cam_dir}: {e}')
    return file_rename_map, file_count


if __name__ == "__main__":
    logger = setup_calibration_video_logger(
        "Video-Processing-Logger",
        format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %(message)s',
        extra_attrs=['task_name', 'detail']
    )
    extra_info = {'frame_number': 42, 'calibration_metric': 0.95}

    logger.debug("Debugging video source.", extra=extra_info)
    logger.info("Calibration successful.", extra=extra_info)
    logger.warning("Minor issue detected in calibration.", extra=extra_info)
    logger.error("Calibration failed.", extra=extra_info)
    logger.critical("Critical error in video correction.", extra=extra_info)

    # Now, the logger should also work without `extra`
    logger.debug("Debugging video source.")
    logger.info("Calibration successful.")
    logger.warning("Minor issue detected in calibration.")
    logger.error("Calibration failed.")
    logger.critical("Critical error in video correction.")
