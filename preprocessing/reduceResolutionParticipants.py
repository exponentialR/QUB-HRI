import os
import shutil
from tqdm import tqdm
import cv2
from utils import reduce_resolution


class ReduceResolution:
    def __init__(self, proj_dir, output_dir, start_participant, end_participant, scale_percent=50):
        self.proj_dir = proj_dir
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.scale_percent = scale_percent

        self.participant_dir_list = [os.path.join(self.proj_dir, f'p{i:02d}') for i in
                                     range(self.start_participant, self.end_participant + 1)]
        self.output_part_list = [os.path.join(output_dir, f'p{i:02d}') for i in
                                 range(self.start_participant, self.end_participant + 1)]
        self.camera_views = ['CAM_LR', 'CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV']

    def process(self):
        for part_dir, out_part_dir in zip(self.participant_dir_list, self.output_part_list):
            os.makedirs(out_part_dir, exist_ok=True)
            for cam_view in self.camera_views:
                cam_dir = os.path.join(part_dir, cam_view)
                out_cam_dir = os.path.join(out_part_dir, cam_view)
                os.makedirs(out_cam_dir, exist_ok=True)
                video_files = [f for f in os.listdir(cam_dir) if f.endswith(('.mp4', '.avi', '.npz'))]
                for vid_file in tqdm(video_files, desc=f'Processing {cam_view} in {os.path.basename(part_dir)}'):
                    vid_path = os.path.join(cam_dir, vid_file)
                    out_vid_path = os.path.join(out_cam_dir, vid_file)
                    if vid_file.endswith(('.mp4', '.avi')):
                        if os.path.exists(out_vid_path):
                            print(f'{out_vid_path} already exists. Skipping...')
                            continue
                        else:
                            reduce_resolution(vid_path, out_vid_path, scale_percent=self.scale_percent)

                    elif vid_file.endswith('.npz'):
                        shutil.copy(os.path.join(cam_dir, vid_file), os.path.join(out_cam_dir, vid_file))

                    else:
                        print(f'Unsupported file format: {vid_file}')

    def reduce_resolution(self, vid_path, out_vid_path):
        pass


if __name__ == '__main__':
    proj_dir = "/media/iamshri/EXTERNAL_USB/Waiting-Data"
    output_dir = "/home/iamshri/Documents/Dataset/QUB-PHEO"
    start_participant = 31
    end_participant = 40
    scale_percent = 45

    processor = ReduceResolution(proj_dir, output_dir, start_participant, end_participant, scale_percent)
    processor.process()

