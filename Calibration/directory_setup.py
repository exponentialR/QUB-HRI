import csv
import json
import os

FOLDERS = ['calibration_videos', 'videos_to_correct', 'corrected_videos']


def load_task_order(csv_path):
    participant_data_tasks = {}
    with open(csv_path, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            participant = row['Participant']
            task_order = [row[f'Task_{i + 1}'] for i in range(5)]
            participant_data_tasks[participant] = task_order
    return participant_data_tasks


class QUBData:
    def __init__(self, project_path):
        super(QUBData, self).__init__()
        self.project_path = os.path.join(project_path, 'PHEO-2.5D')
        self.participants = [os.path.join(self.project_path, 'Participants', f'P{p:02d}') for p in range(0, 100)]
        self.camera_positions = ['CAM_LL', 'CAM_UL', 'CAM_AV', 'CAM_UR', 'CAM_LR']
        self.task_video_counts = {'BIAH': 3, 'BRIDGE': 2, 'CALIBRATION': 1, 'STAIRWAY': 2, 'TOWER': 2}
        self.skipped_data = {f'p{i:02d}': {} for i in range(100)}

    def create_participant_repository(self):
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
            print(f'CREATED DATA PROJECT REPOSITORY {self.project_path}')
        else:
            print(f'Project Repository {self.project_path} EXISTS Please use another Name')
            exit()
        _ = [print(f'CREATED SUBFOLDER {participant}') and os.makedirs(participant) if not os.path.exists(
            participant) else print(f'{participant} EXISTS') for participant in self.participants]

    def create_meta_data(self, participant_info, task_info, camera_info):
        meta_data = {
            'participant_info': participant_info,
            'task_info': task_info,
            'camera_info': camera_info
        }
        meta_data_path = os.path.join(self.project_path, 'metadata.json')

        with open(meta_data_path, 'w') as f:
            json.dump(meta_data, f, indent=4)

        print(f'METADATA SAVED AT {meta_data_path}')

    def rename_participant_files(self, csv_path):
        participant_data_tasks = load_task_order(csv_path)
        for participant_data in self.participants:
            current_participant = os.path.basename(participant_data)
            if os.path.isdir(participant_data):
                for camera_position in self.camera_positions:
                    camera_specific_data = os.path.join(participant_data, camera_position)
                    video_list = [i for i in os.listdir(camera_specific_data) if i.endswith('.MP4')]
                    if len(video_list) == 10:
                        video_list = sorted(video_list)
                        video_index = 0
                        for task in participant_data_tasks[current_participant]:
                            for version_index in range(self.task_video_counts[task]):
                                old_path = os.path.join(camera_specific_data, video_list[video_index])
                                version_name = self.task_video_counts[task][version_index]
                                new_filename = f'{task}_{version_name}_{camera_position}.MP4'
                                new_path = os.path.join(camera_specific_data, new_filename)
                                os.rename(old_path, new_path)
                                print(f'Renamed {old_path} to {new_path}')
                                video_index += 1
                    elif len(video_list) < 10:
                        self.skipped_data[current_participant][f'{camera_position}'] = '<10'
                        print(
                            f'Skipped {camera_position} for Participant {current_participant} due to Video files < 10... Please check')


                    else:
                        self.skipped_data[current_participant][camera_position] = '>10'
                        print(
                            f'SKipped {camera_position} for Participant {current_participant} due to Video files > 10... Please Check ')
            else:
                print(f'Participant Directory {current_participant} does not exist')



