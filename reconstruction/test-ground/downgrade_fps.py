import os
import cv2
import numpy as np
from tqdm import tqdm


def downgrade_fps(left_video, right_video, new_video_path=None):
    """
    Downgrade the fps of the video to the fps of the guiding video
    The guiding video is the video with the lower fps
    :param left_video: str: Path to the left video
    :param right_video: str: Path to the right video
    :param new_video_path: str: Path to save the new video
    :return: None
    """
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    print(f'Right FPS: {right_fps} \n Left FPS: {left_fps}')

    if left_fps > right_fps:
        target_fps = right_fps
        higher_fps_cap = left_cap
        video_to_downgrade = left_video
        skip_rate = left_fps / right_fps

    else:
        target_fps = left_fps
        higher_fps_cap = right_cap
        video_to_downgrade = right_video
        skip_rate = right_fps / left_fps

    print(f'Video to downgrade: {video_to_downgrade} \n Target FPS: {target_fps} \n Skip Rate: {skip_rate}')

    if new_video_path is None:
        new_video_path = os.path.join(os.path.dirname(video_to_downgrade),
                                      os.path.basename(video_to_downgrade).split('.')[0] + '_downgraded.MP4')

    total_frames = int(higher_fps_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(higher_fps_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(higher_fps_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    out = cv2.VideoWriter(new_video_path, fourcc, target_fps, (frame_width, frame_height))

    frame_id = 0
    frames_written = 0
    with tqdm(total=total_frames // skip_rate, desc=f'Downgrading FPS and Saving to {new_video_path}') as pbar:
        while True:
            ret, frame = higher_fps_cap.read()
            if not ret:
                break

            # Write every nth frame to match the target fps
            if frame_id % skip_rate < 1:
                out.write(frame)
                frames_written += 1
                pbar.update(1)

            frame_id += 1

    print(f"Frames written: {frames_written}")


if __name__ == '__main__':
    left_video_path = '/home/iamshri/Documents/Test-Video/p03/CAM_LL/CALIBRATION_CC.MP4'
    right_video_path = '/home/iamshri/Documents/Test-Video/p03/CAM_LR/CALIBRATION_CC.MP4'
    downgrade_fps(left_video_path, right_video_path)
