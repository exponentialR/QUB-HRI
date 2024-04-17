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


if __name__ == "__main__":
    # Example setup
    logger = setup_calibration_video_logger(
        "Renaming-Videos-Logger",
        format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %(message)s',
        extra_attrs=['task_name', 'detail'],
        error_log_file='test.txt',
        levels_to_save={logging.DEBUG, logging.INFO},  # Set levels to save
        console_level=logging.INFO  # Set console level
    )

    # Test logs
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
