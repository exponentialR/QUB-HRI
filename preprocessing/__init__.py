"""Preprocessing helpers; expensive legacy dependencies load only when used."""

__all__ = ["downgrade_fps", "match_frame_length", "setup_calibration_video_logger"]


def __getattr__(name):
    if name in {"downgrade_fps", "match_frame_length"}:
        from reconstruction import downgrade_fps as module

        return getattr(module, name)
    if name == "setup_calibration_video_logger":
        from .utils import setup_calibration_video_logger

        return setup_calibration_video_logger
    raise AttributeError(name)
