import os.path
import os
import numpy as np
import cv2
from tqdm import tqdm


def extract_name(full_path):
    components = full_path.split('/')
    desired_path = ''
    for component in components:
        if component.startswith('p') or component.startswith('CAM_') or component.endswith('.mp4'):
            desired_path = os.path.join(desired_path, component)
    return desired_path


def downgrade_fps(left_video, right_video, new_video_path=None, logger=None):
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
    # logger.info(f'Right FPS: {right_fps} \n Left FPS: {left_fps}')

    if left_fps > right_fps:
        target_fps = right_fps
        higher_fps_cap = left_cap
        low_fps_video = right_video
        video_to_downgrade = left_video
        skip_rate = left_fps / right_fps

    elif left_fps < right_fps:
        target_fps = left_fps
        higher_fps_cap = right_cap
        low_fps_video = left_video
        video_to_downgrade = right_video
        skip_rate = right_fps / left_fps
    else:
        logger.info(f'{extract_name(left_video)} and {extract_name(right_video)} have the same FPS. No need to downgrade', extra={'task_name': 'Downgrade FPS', 'detail': 'FPS is the same'})
        return None

    logger.info(f'Video to downgrade: {extract_name(video_to_downgrade)} \n Target FPS: {target_fps} \n Skip Rate: {skip_rate}')

    if new_video_path is None:
        new_video_path = os.path.join(os.path.dirname(video_to_downgrade),
                                      os.path.basename(video_to_downgrade).split('.')[0] + '_downgraded.MP4')

    total_frames = int(higher_fps_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(higher_fps_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(higher_fps_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

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
    if new_video_path is not None:
        # Delete the original video
        os.remove(video_to_downgrade)
        logger.info(f"Original video deleted: {video_to_downgrade}", extra={'task_name': 'Downgrade FPS', 'detail': 'Original video deleted'})

        os.rename(new_video_path, video_to_downgrade)

        # print(f"Frames written: {frames_written}")
        higher_fps_cap.release()
        out.release()
        cv2.destroyAllWindows()
        left_cap.release()
        right_cap.release()

    return video_to_downgrade


def match_frame_length(left_video_addr, right_video_addr, logger):
    left_cap = cv2.VideoCapture(left_video_addr)
    right_cap = cv2.VideoCapture(right_video_addr)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    left_cap_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_cap_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # logger.info(f'Frame count ==> {os.path.basename(left_video_addr)} = {left_cap_count} | {os.path.basename(right_video_addr)} = {right_cap_count}')
    # print(f'Total Frames in Right Video: {right_cap_count}')

    if left_cap_count == right_cap_count:
        logger.info(
            f'Both videos {extract_name(right_video_addr)} and {extract_name(left_video_addr)} have the same number of frames', extra={'task_name': 'Match Frame Length', 'detail': 'Frame count is the same'})
        # print(f'FRAME COUNT: {left_cap_count}')
        return None, None

    elif left_cap_count < right_cap_count:
        # print(f'Left video has fewer frames than the low fps video')
        n_frames_to_keep = left_cap_count
        trimmed_video_path = os.path.join(os.path.dirname(right_video_addr),
                                          os.path.basename(right_video_addr).split('.')[0] + '_trimmed.MP4')
        video_addr = right_video_addr
        video_to_trim_cap = right_cap
        target_fps = right_cap.get(cv2.CAP_PROP_FPS)
        frame_width, frame_height = int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(trimmed_video_path, fourcc, target_fps, (frame_width, frame_height))
        logger.info(
            f'{right_cap_count - left_cap_count} frames will be trimmed from {extract_name(right_video_addr)}')

    else:
        # print(f'Right video has fewer frames than the left video')
        n_frames_to_keep = right_cap_count
        trimmed_video_path = os.path.join(os.path.dirname(left_video_addr),
                                          os.path.basename(left_video_addr).split('.')[0] + '_trimmed.MP4')
        video_addr = left_video_addr
        video_to_trim_cap = left_cap
        target_fps = left_cap.get(cv2.CAP_PROP_FPS)
        frame_width, frame_height = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(trimmed_video_path, fourcc, target_fps, (frame_width, frame_height))
        logger.info(
            f'{left_cap_count - right_cap_count} frames will be trimmed from {extract_name(left_video_addr)}', extra={'task_name': 'Match Frame Length', 'detail': 'Trimming video'})

    frame_count = 0
    while frame_count < n_frames_to_keep:
        ret, frame = video_to_trim_cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    video_to_trim_cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.remove(video_addr)
    os.rename(trimmed_video_path, video_addr)
    # logger.info(f"Trimmed video now saved to: {video_addr}")

    left_count, right_count = compare_frame_count(left_video_addr, right_video_addr, logger)
    if left_count == right_count:
        logger.info(f'Both {extract_name(left_video_addr)} and {extract_name(right_video_addr)} now have the same number of frames', extra={'task_name': 'Match Frame Length', 'detail': f'Frame count :{left_count}'})
    else:
        logger.warning(f'Both {extract_name(left_video_addr)} and {extract_name(right_video_addr)} do not have the same number of frames', extra={'task_name': 'Match Frame Length', 'detail': f'Frame count :{left_count} | {right_count}'})
        # Attempt to trim the videos again
        match_frame_length(left_video_addr, right_video_addr, logger)
    left_cap.release()
    right_cap.release()

    return left_video_addr, right_video_addr

    # Compare the frames to confirm that the videos have the same number of frames


def compare_frame_count(left_video, right_video, logger):
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # logger.info(f'{os.path.basename(left_video)}:{left_frame_count} | {os.path.basename(right_video)}:{right_frame_count}')
    left_cap.release(), right_cap.release()
    return left_frame_count, right_frame_count


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    # logging.getLogger('moviepy').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    left_video_path = '/home/qub-hri/Documents/Datasets/QUB-PHEO/p61/CAM_AV/TOWER_MS.mp4'
    right_video_path = '/home/qub-hri/Documents/Datasets/QUB-PHEO/p61/CAM_AV_P/TOWER_MS.mp4'
    downgraded_video = downgrade_fps(left_video_path, right_video_path, logger=logger)
    match_frame_length(left_video_path, right_video_path, logger)
