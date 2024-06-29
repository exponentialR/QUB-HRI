import os
from pathlib import Path

CAM_VIEWS = ['CAM_AV', 'CAM_LL', 'CAM_LR', 'CAM_UR', 'CAM_UL']


def check_task_presence(project_dir, start_part, end_part, task_to_check='calib', file_ext=(('mp4', 'avi'))):
    file_presence_count = 0
    for participant in range(start_part, end_part + 1):
        participant_id = f'p{participant:02d}'

        for cam_view in CAM_VIEWS:
            original_task_path = os.path.join(project_dir, participant_id, cam_view)
            file_check_list = [file for file in os.listdir(original_task_path) if file.lower().startswith(task_to_check) and file.lower().endswith(file_ext)]
            file_presence_count +=len(file_check_list)
            print(f'{file_presence_count} {task_to_check} Present in {participant_id}/{cam_view}')
        print(f'{file_presence_count} {task_to_check} Present in {participant_id}')
    print(f'{file_presence_count} {task_to_check} Present in Participants p{start_part:02d} to p{end_part:02d}')


if __name__ == '__main__':
    start_part, end_part = 1, 70
    PROJECT_DIR = '/media/BlueHDD/Raw-QUB-PHEO/uncorrected'

    check_task_presence(project_dir=PROJECT_DIR, start_part=start_part, end_part=end_part)



