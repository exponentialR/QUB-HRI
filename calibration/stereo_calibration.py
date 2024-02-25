import cv2
import numpy as np
from transfer_calibration import copy_calibration_videos
from extrinsic_calibration import ExtrinsicCalibration


class StereoPipeline:
    def __init__(self, start_participant, end_participant, original_project_path, project_dir):
        self.start_participant, self.end_participant = start_participant, end_participant
        self.original_project_path, self.project_dir = original_project_path, project_dir
        copy_calibration_videos(self.project_dir, self.start_participant, self.end_participant,
                                self.original_project_path)

    def run(self):
        """
        Run stereo calibration for all participants in the project directory
        This function assumes that the original calibration videos have been transferred to the project directory
        And that the individual camera intrinsic calibration data has been obtained
        """
        for participant_id in range(self.start_participant, self.end_participant + 1):
            left_calib_path = f'{self.project_dir}/p{participant_id:02d}/CAM_LL/calib_param_CALIBRATION.npz'
            right_calib_path = f'{self.project_dir}/p{participant_id:02d}/CAM_LR/calib_param_CALIBRATION.npz'
            left_video_path = f'{self.project_dir}/p{participant_id:02d}/CAM_LL/Original_CALIBRATION.MP4'
            right_video_path = f'{self.project_dir}/p{participant_id:02d}/CAM_LR/Original_CALIBRATION.MP4'
            extrinsic_calib = ExtrinsicCalibration(left_calib_path, right_calib_path, left_video_path, right_video_path)
            stereo_data_path = extrinsic_calib.run()
            print(f'Stereo calibration data for p{participant_id:02d} saved to {stereo_data_path}')
        pass


if __name__ == '__main__':
    project_dir = '/media/iamshri/Seagate/QUB-PHEOVision'
    original_project_path = '/home/iamshri/Documents/PHEO-Data/Unprocessed'
    start_participant = 1
    end_participant = 2
    stereo_pipeline = StereoPipeline(start_participant, end_participant, original_project_path, project_dir)
    stereo_pipeline.run()
