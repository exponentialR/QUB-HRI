import cv2
import os
from tqdm import tqdm


import cv2
import os
from tqdm import tqdm

def extract_frame(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a specified video file at a defined interval and saves them as images
    in a given output folder. The frame file names are prefixed with the video file name.

    Args:
        video_path (str): The full path to the video file.
        output_folder (str): The directory where extracted frames will be stored.
        frame_interval (int): Interval between frames to extract, e.g., `10` means every 10th frame is saved.

    Creates:
        A series of image files representing selected frames of the video, stored in the specified output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Error: Could not open video {video_path}')
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames // frame_interval, desc=f"Extracting frames from {video_name}")

    frame_count = 0
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame_filename = f'{video_name}_{frame_count:04d}.jpg'
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            pbar.update(1)

        frame_index += 1

    cap.release()
    pbar.close()
    print(f'Frames extracted: {frame_count} from {video_path}')


if __name__ == '__main__':
    """
    Main execution block that processes multiple video files in a directory,
    extracting frames at a specified interval from each video and storing them in
    separate subdirectories for each video file.
    """
    video_directory = '/media/samueladebayo/Expansion/PHEO/object_detection'
    output_directory = '/home/samueladebayo/Documents/Projects/QUB-PHEO/Object-DetectionDataset'
    prename = 'frame'
    frame_interval = 5

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all video files in the directory
    video_files = [f for f in os.listdir(video_directory) if f.endswith(('MP4', '.mp4', '.avi', '.mov'))]
    total_videos = len(video_files)
    video_progress = tqdm(video_files, desc="Processing videos")

    for video_file in video_progress:
        video_path = os.path.join(video_directory, video_file)
        # Create a unique folder for each video's frames
        specific_output_folder = os.path.join(output_directory, os.path.splitext(video_file)[0])
        extract_frame(video_path, specific_output_folder, frame_interval)