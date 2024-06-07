import os
import shutil

FILE_EXTS = ('avi', 'mp4')


def copy_like_files(project_dir, start_participant, end_participant, output_folder, tasks_to_copy,
                    camera_view='CAM_AV_P'):
    for participant in range(start_participant, end_participant + 1):
        participant = f'p{participant:02d}'
        output_participant_path = os.path.join(output_folder, tasks_to_copy)
        os.makedirs(output_participant_path) if not os.path.exists(output_participant_path) else None
        participant_cam_view = os.path.join(project_dir, participant, camera_view)
        list_files = [file for file in os.listdir(participant_cam_view) if
                      file.endswith((FILE_EXTS)) and file.startswith(tasks_to_copy)]
        for video_file in list_files:
            output_path = os.path.join(output_participant_path, f'{participant}_{camera_view}_{video_file}')
            original_video_file = os.path.join(participant_cam_view, video_file)

            if not os.path.exists(original_video_file):
                print(f'FILE {original_video_file} DOES NOT EXISTS')
            else:
                shutil.copy(original_video_file, output_path)


if __name__ == '__main__':
    project_dir = '/home/samueladebayo/Documents/Dataset/QUB-PHEO/'
    start_participant = 7
    end_participant = 7
    output_folder = '/home/samueladebayo/Documents/Dataset/transfer_like_tasks'
    tasks_to_copy = 'STAIRWAY_AP'
    camera_view = 'CAM_AV_P'
    copy_like_files(project_dir, start_participant, end_participant, output_folder, tasks_to_copy, camera_view)
