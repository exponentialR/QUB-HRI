import os
import shutil
import json
from utils import process_cam_directory


class RenameVideos:
    def __init__(self, proj_dir, start_participant, end_participant, task_metadata='tasks.json'):
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.camera_views = ['CAM_LR', 'CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV']
        self.participant = [os.path.join(proj_dir, f'p{i:02d}') for i in
                            range(self.start_participant, self.end_participant+1)]
        with open(task_metadata, 'r') as file:
            self.task_names = json.load(file)
        if not os.path.exists('file_renamed.json'):
            with open('file_renamed.json', 'w') as file:
                json.dump({}, file)
        with open('file_renamed.json', 'r') as file:
            self.file_renamed = json.load(file)

    def rename(self):
        total_video_count = 0
        for participant_id_dir in self.participant:
            part_id = os.path.basename(participant_id_dir)
            cam_rename_count = 0
            for cam_view in self.camera_views:
                cam_dir = os.path.join(participant_id_dir, cam_view)
                if not os.path.exists(cam_dir):
                    print(f"Directory {cam_dir} not found.")
                    continue
                if part_id not in self.file_renamed:
                    self.file_renamed[part_id] = {}
                if cam_view not in self.file_renamed[part_id]:
                    self.file_renamed[part_id][cam_view] = {}

                task_metadata = self.task_names[part_id]
                file_rename_map, file_count = process_cam_directory(cam_dir, task_metadata)
                if file_count is not None:
                    self.file_renamed[part_id][cam_view] = file_rename_map
                    cam_rename_count += file_count
            print(f'{cam_rename_count} files renamed in {part_id}')
            total_video_count += cam_rename_count
        with open('file_renamed.json', 'w') as file:
            json.dump(self.file_renamed, file, indent=6)
        print(
            f'Total {total_video_count} files renamed between p{self.start_participant:02d} and p{self.end_participant:02d}')


if __name__ == '__main__':
    proj_dir = "/home/iamshri/Documents/Dataset/Test_Evironment"
    start_participant = 16
    end_participant = 20
    task_metadata = 'tasks.json'

    renamer = RenameVideos(proj_dir, start_participant, end_participant, task_metadata)
    renamer.rename()
