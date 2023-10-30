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

class CalibrationVideoFormatter(logging.Formatter):
    def format(self, record):
        record.frame_number = record.frame_number if hasattr(record, 'frame_number') else 'N/A'
        record.calibration_metric = record.calibration_metric if hasattr(record, 'calibration_metric') else 'N/A'
        log_message = super().format(record)
        return f"{COLOURS[record.levelname]}{log_message}{COLOURS['ENDC']}"

def setup_calibration_video_logger():
    # Create logger
    logger = logging.getLogger("Video-Calibration-Correction")
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = CalibrationVideoFormatter(
        '%(asctime)s - %(name)s - [Frame: %(frame_number)s] - [Calibration Metric: %(calibration_metric)s] - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    logger = setup_calibration_video_logger()

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
