import csv
from datetime import datetime
from pathlib import Path
import os
from utils.logger_utils import setup_calibration_video_logger

logger = setup_calibration_video_logger()


def rearrange_pheo(parent_dir, start_participant, last_participant):
    """
    Renames video files in specific directories to 'CALIBRATION.MP4' based on certain conditions.

    Parameters:
    - parent_dir (str): The parent directory containing participant folders.
    - start_participant (int): The ID of the first participant to start the rearrangement from.
    - last_participant (int): The ID of the last participant for the rearrangement.

    Behavior:
    - Iterates through participant folders specified by the range [start_participant, last_participant].
    - In each participant folder, looks for specific camera view folders ['CAM_AV', 'CAM_LL', 'CAM_UR', 'CAM_LR', 'CAM_UL'].
    - In each camera view folder, renames MP4 files to 'CALIBRATION.MP4' if there are 10 or fewer MP4 files and the first one is not already named 'CALIBRATION.MP4'.
    """

    # Get the current date and time for unique file naming
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create or open a log file to log renamed files
    with open(f'{current_time}_renamed_files.log', 'w') as renamed_file:
        # Create or open a log file to log directories with more than 10 videos
        with open(f'{current_time}_more_than_10_videos.log', 'w') as more_than_10_file:
            # Write headers to log files
            renamed_file.write("Original File, Renamed File\n")
            more_than_10_file.write("Directory\n")

            participant_list = [os.path.join(parent_dir, f'{participant_id:02d}') for participant_id in
                                range(start_participant, last_participant + 1)]

            # Loop through each participant folder
            for participant_id in participant_list:
                current_participant = os.path.join(parent_dir, participant_id)

                # Define the camera view folders to look for in each participant folder
                camera_views = [os.path.join(current_participant, camera_view) for camera_view in
                                ['CAM_AV', 'CAM_LL', 'CAM_UR', 'CAM_LR', 'CAM_UL']]

                # Loop through each camera view folder
                for camera_view in camera_views:
                    # List MP4 files in the current camera view folder
                    camera_view_videos = [i for i in sorted(os.listdir(camera_view)) if i.endswith('MP4')]

                    # Check if there are 10 or fewer MP4 files
                    if len(camera_view_videos) <= 10:

                        # Skip if the first video is already named 'CALIBRATION.MP4'
                        if camera_view_videos[0] == 'CALIBRATION.MP4':
                            pass
                        else:
                            # Generate full file paths for each video file
                            camera_view_videos = [os.path.join(camera_view, video_file) for video_file in camera_view_videos]

                            # Loop through each video file to rename it
                            for original_video_path in camera_view_videos:
                                new_video_filepath = os.path.join(camera_view, 'CALIBRATION.MP4')

                                # Check if the original video file exists
                                if os.path.exists(original_video_path):

                                    # Check if a file with the new name already exists
                                    if not os.path.exists(new_video_filepath):
                                        os.rename(original_video_path, new_video_filepath)
                                        logger.info(f'Renamed {original_video_path} to {new_video_filepath}')
                                        renamed_file.write(f"{original_video_path}, {new_video_filepath}\n")

                                    else:
                                        print(f"A file already exists at {new_video_filepath}. Rename operation aborted.")
                                else:
                                    print(f"The file {original_video_path} does not exist.")
                    else:
                        logger.critical(f'Video Files more than 10 Please check {camera_view}')
                        more_than_10_file.write(f"{camera_view}\n")


if __name__ == '__main__':
    parent_dir = ''
    rearrange_pheo(parent_dir, 1, 10)