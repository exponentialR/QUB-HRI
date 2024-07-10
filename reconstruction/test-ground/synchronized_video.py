import cv2
import numpy as np
import os


def frame_difference(left_video, right_video):
    """
    Sunchronize two videos by skipping the first few frames of the longer video.
    Then Visualize them size by side.
    :param left_video: str: Path to the left video
    :param right_video: str: Path to the right video
    return: start_frame: int: The frame to start reading for the longer video and which video is longer
    """
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_difference = abs(left_frame_count - right_frame_count)
    print(f'There are {frame_count_difference} frames difference between the two videos')
    print(
        f'attempting to synchronize videos {os.path.basename(left_video)} and {os.path.basename(right_video)}')
    print(f'Left video frame count: {left_frame_count}')
    print(f'Right video frame count: {right_frame_count}')
    if left_frame_count > right_frame_count:
        print(f'Skip frame {frame_count_difference} in left video')
        longer_video = 'left'
    elif right_frame_count > left_frame_count:
        print(f'Skip frame {frame_count_difference} in right video')
        longer_video = 'right'
    else:
        longer_video = 'none'
    return frame_count_difference, longer_video


def display_synchronized_video(left_video, right_video, frame_count_diff, video_long):
    """
    Display the synchronized videos side by side by starting from the start_frame of the longer video.
    :param left_video: str: Path to the left video
    :param right_video: str: Path to the right video
    :param frame_count_diff: int: The frame to start reading for the longer video
    :param video_long: str: The longer video
    :return: None
    """
    print(frame_count_diff, video_long)
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)

    # Set the starting frame for the longer video
    if video_long == 'left':
        print(f'Skipping {frame_count_diff} frames in left video')
        for _ in range(frame_count_diff):
            left_cap.read()  # Skip initial frames in left video
        # left_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    elif video_long == 'right':
        print(f'Skipping {frame_count_diff} frames in right video')
        for _ in range(frame_count_diff):
            right_cap.read()  # Skip initial frames in right video
    elif video_long == 'none':
        print('Both videos have the same number of frames')
        return
        # right_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    common_fps = min(left_fps, right_fps)

    while True:
        left_time = left_cap.get(cv2.CAP_PROP_POS_MSEC)
        right_time = right_cap.get(cv2.CAP_PROP_POS_MSEC)
        time_difference = abs(left_time - right_time)

        if time_difference > (1000 / common_fps):
            if left_time < right_time:
                left_cap.read()
            else:
                right_cap.read()
            continue
        ret_left, frame_left = left_cap.read()
        ret_right, frame_right = right_cap.read()

        # Break the loop if either video ends
        if not ret_left or not ret_right:
            print("Reached the end of one or both videos.")
            break

        # Resize the frames to have the same height
        height_left, width_left = frame_left.shape[:2]
        height_right, width_right = frame_right.shape[:2]
        height = 920  # min(height_left, height_right)  # Use the smaller height for both frames
        frame_left = cv2.resize(frame_left, (int(width_left * (height / height_left)), height))
        frame_right = cv2.resize(frame_right, (int(width_right * (height / height_right)), height))

        # Combine the frames side by side
        combined_frame = np.hstack((frame_left, frame_right))

        # Display the combined frame
        cv2.imshow('Synchronized Videos', combined_frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture objects and close all windows
    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()


def display_video(left_video, right_video):
    """
    Display the synchronized videos side by side by starting from the start_frame of the longer video.
    :param left_video: str: Path to the left video
    :param right_video: str: Path to the right video
    :return: None
    """
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)

    while True:
        ret_left, frame_left = left_cap.read()
        ret_right, frame_right = right_cap.read()

        # Break the loop if either video ends
        if not ret_left or not ret_right:
            print("Reached the end of one or both videos.")
            break

        # Resize the frames to have the same height
        height_left, width_left = frame_left.shape[:2]
        height_right, width_right = frame_right.shape[:2]
        height = 920  # min(height_left, height_right)  # Use the smaller height for both frames
        frame_left = cv2.resize(frame_left, (int(width_left * (height / height_left)), height))
        frame_right = cv2.resize(frame_right, (int(width_right * (height / height_right)), height))

        # Combine the frames side by side
        combined_frame = np.hstack((frame_left, frame_right))

        # Display the combined frame
        cv2.imshow('Synchronized Videos', combined_frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture objects and close all windows
    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()


# Example usage
left_video_path = '/home/iamshri/Documents/Dataset/p01/CAM_LR/BIAH_BS.mp4'
right_video_path = '/home/iamshri/Documents/Dataset/p01/CAM_LL/BIAH_BS.mp4'
start_frame, longer_video = frame_difference(left_video_path, right_video_path)
# print(f' Start Frame: {start_frame}', f' Longer Video: {longer_video}')
display_synchronized_video(left_video_path, right_video_path, start_frame, longer_video)
display_video(left_video_path, right_video_path)
