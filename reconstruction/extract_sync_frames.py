import os.path
import os
import numpy as np
import cv2
from utils import create_dir, remove_files_in_folder, detect_charuco_board


def extract_synchronized_frames(left_video_path, right_video_path, aruco_dict, board, frame_interval=1, min_corners=5,
                                max_frames=1000):
    save_left_path = os.path.join(os.path.dirname(left_video_path), 'StereoCalibFrames')
    save_right_path = os.path.join(os.path.dirname(right_video_path), 'StereoCalibFrames')
    create_dir(save_right_path)
    create_dir(save_left_path)
    if len(os.listdir(save_left_path)) >= max_frames or len(os.listdir(save_right_path)) >= max_frames:
        print(f'{max_frames} Frames have already been extracted from {os.path.basename(left_video_path)} and '
              f'{os.path.basename(right_video_path)}. Skipping...')
        return save_left_path, save_right_path

    if len(os.listdir(save_left_path)) < max_frames or len(os.listdir(save_right_path)) > max_frames:
        remove_files_in_folder(save_left_path)
        remove_files_in_folder(save_right_path)

    cap_left = cv2.VideoCapture(left_video_path)
    cap_right = cv2.VideoCapture(right_video_path)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print(f'Could not open one or both video files')
        return

    left_frame_count = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frame_count = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_difference = abs(left_frame_count - right_frame_count)
    print(f'There are {frame_count_difference} frames difference between the two videos')
    print(
        f'attempting to synchronize videos {os.path.basename(left_video_path)} and {os.path.basename(right_video_path)}')

    if left_frame_count > right_frame_count:
        for _ in range(frame_count_difference):
            cap_left.read()  # Skip initial frames in left video
        print(f'Skip frame {frame_count_difference} in left video')
    elif right_frame_count > left_frame_count:
        for _ in range(frame_count_difference):
            cap_right.read()  # Skip initial frames in right video
        print(f'Skip frame {frame_count_difference} in right video')

    frame_count = 0
    saved_frame_count = 0

    left_fps = cap_left.get(cv2.CAP_PROP_FPS)
    right_fps = cap_right.get(cv2.CAP_PROP_FPS)
    common_fps = min(left_fps, right_fps)

    while saved_frame_count < max_frames:
        left_time = cap_left.get(cv2.CAP_PROP_POS_MSEC)
        right_time = cap_right.get(cv2.CAP_PROP_POS_MSEC)
        time_difference = abs(left_time - right_time)

        if time_difference > (1000 / common_fps):
            if left_time < right_time:
                cap_left.read()
            else:
                cap_right.read()
            continue

        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print(
                f'Done processing videos {os.path.basename(left_video_path)} and {os.path.basename(right_video_path)}')
            break

        if frame_count % frame_interval == 0:
            visible_left = detect_charuco_board(frame_left, aruco_dict, board, min_corners)
            visible_right = detect_charuco_board(frame_right, aruco_dict, board, min_corners)

            if visible_left and visible_right:
                cv2.imwrite(os.path.join(save_left_path, f"frame_{frame_count:04d}.png"), frame_left)
                cv2.imwrite(os.path.join(save_right_path, f"frame_{frame_count:04d}.png"), frame_right)
                saved_frame_count += 1
                print(f'Saved frame pair {saved_frame_count}')
        frame_count += 1
    if saved_frame_count < max_frames:
        print(f'Could not extract enough frames for stereo calibration. Extracted {saved_frame_count} frames')
    cap_left.release()
    cap_right.release()
    return save_left_path, save_right_path
