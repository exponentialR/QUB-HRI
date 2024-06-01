import os

FILE_EXTS = ['avi', 'mp4']


def copy_like_files(project_dir, start_participant, end_participant, output_folder, tasks_to_copy,
                    camera_view='CAM_AV_P'):
    participants_dir = [(os.path.join(project_dir, f'p{participant:02d}', camera_view), f'p{participant:02d}') for
                        participant in range(start_participant, end_participant + 1)]

    for participant_dir, participant in participants_dir:
        files_in_participant_dir = [f for f in os.listdir(participant_dir) if
                                    os.path.isfile(os.path.join(participant_dir, f)) and f.lower().endswith((FILE_EXTS))
                                    and tasks_to_copy in f]
        print(files_in_participant_dir)


if __name__ == '__main__':
    project_dir = '/home/iamshri/Documents/Dataset'
    start_participant = 1
    end_participant = 1
    output_folder = '/home/iamshri/Documents/Dataset/transfer_like_tasks'
    tasks_to_copy = 'BIAH'
    camera_view = 'CAM_AV_P'
    copy_like_files(project_dir, start_participant, end_participant, output_folder, tasks_to_copy, camera_view)
