"""
Landmark Extraction Module

This module is designed to process video files for specific participants to extract and store facial, pose, and hand landmarks using MediaPipe. The extracted landmarks are saved in an HDF5 format for subsequent analysis. This process is vital for research fields where detailed human movement and feature analysis are required.

License:
Copyright (C) 2024 Samuel Adebayo
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Dependencies:
- cv2 (opencv-python)
- mediapipe
- h5py
- numpy
- tqdm
- custom logging setup from `utils`

Usage:
- Configure the project directory, start and end participant indices, and ensure all dependencies are installed.
- Run the script to process videos from specified camera views for each participant, extracting and saving the landmarks.

Author: Samuel Adebayo
Year: 2024

"""


from reconstruction.sideview_keyextraction import LandmarksToHDF5
import os
from tqdm import tqdm
from reconstruction.utils import setup_calibration_video_logger
import logging

logger = setup_calibration_video_logger(
    "Landmark-Extraction-Logger",
    format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %('
               'message)s',
    extra_attrs=['task_name', 'detail'],
    error_log_file='landmark_extraction_log.txt',
    levels_to_save={logging.DEBUG, logging.INFO},
    console_level=logging.INFO
)


class ExtractLandmark:
    def __init__(self, proj_directory, start_part, end_part, logger=None, track_confidence=0.5,
                 detect_confidence=0.5):
        self.start_participant = start_part
        self.end_participant = end_part
        self.logger = logger
        self.proj_dir = proj_directory
        self.participant_dir_list = [os.path.join(self.proj_dir, f'p{str(i).zfill(2)}') for i in
                                     range(self.start_participant, self.end_participant + 1)]
        self.camera_views = ['CAM_LL', 'CAM_LR']
        self.track_confidence = track_confidence
        self.detect_confidence = detect_confidence

    def extract_landmarks(self):
        total_count_landmark = 0
        for participant_dir in tqdm(self.participant_dir_list, desc="Participant Progress"):
            participant_dir_str = os.path.basename(participant_dir)
            logger.info(f'Processing participant {participant_dir_str}',
                        extra={'task_name': 'Landmark Extraction', 'detail': 'Participant Processing'})

            part_count_landmark = 0
            for camera_view in self.camera_views:
                camera_view_dir = os.path.join(participant_dir, camera_view)
                video_list = sorted(
                    [os.path.join(camera_view_dir, video_file) for video_file in os.listdir(camera_view_dir)
                     if video_file.endswith('.mp4') and not video_file.startswith('calib')])
                landmark_count = 0
                for video_path in tqdm(video_list, desc="Video Progress"):
                    video_name = os.path.basename(video_path)[:os.path.basename(video_path).rfind('.')]

                    hdf5_dir = os.path.join(camera_view_dir, 'landmarks')
                    if not os.path.exists(hdf5_dir):
                        os.makedirs(hdf5_dir, exist_ok=True)

                    hdf5_path = os.path.join(hdf5_dir, f'{video_name}.hdf5')
                    logger.info(f'Processing video {participant_dir_str}/{camera_view}/{video_name}',
                                extra={'task_name': 'Landmark Extraction', 'detail': 'Video Processing'})
                    processor = LandmarksToHDF5(video_path, hdf5_path, self.detect_confidence, self.track_confidence)
                    processor.process_and_save()
                    landmark_count += 1

                logger.info(f'Processed {landmark_count} videos in {participant_dir_str}/{camera_view}',
                            extra={'task_name': 'Landmark Extraction', 'detail': 'Video Processing'})
                part_count_landmark += landmark_count
            logger.info(f'Processed {part_count_landmark} videos in {participant_dir_str}', extra={
                'task_name': 'Landmark Extraction', 'detail': 'Participant Processing'})
            total_count_landmark += part_count_landmark
        logger.info(
            f'{total_count_landmark} hdf5 landmarks saved for participants p{self.start_participant:02d} to p{self.end_participant:02d}',
            extra={'task_name': 'Landmark Extraction', 'detail': 'Total Processing'})


if __name__ == '__main__':
    proj_dir = '/home/iamshri/Documents/Dataset'
    start_participant = 1
    end_participant = 1
    landmark_extractor = ExtractLandmark(proj_dir, start_participant, end_participant, logger=logger)
    landmark_extractor.extract_landmarks()
