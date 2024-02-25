import os
import shutil


def copy_calibration_videos(project_dir, start_participant, end_participant, original_project_path):
    camera_views = ['CAM_AV', 'CAM_LL', 'CAM_UR', 'CAM_LR', 'CAM_UL']
    directory_copy_count = 0
    for participant_id in range(start_participant, end_participant + 1):
        calib_part_count = 0
        for camera_view in camera_views:
            original_video_path = os.path.join(original_project_path, f'p{participant_id:02d}', camera_view,
                                               'CALIBRATION.MP4')
            new_video_filepath = os.path.join(project_dir, f'p{participant_id:02d}', camera_view, 'Original_CALIBRATION.MP4')
            if os.path.exists(original_video_path):
                if not os.path.exists(new_video_filepath):
                    os.makedirs(os.path.dirname(new_video_filepath), exist_ok=True)
                    shutil.copy(original_video_path, new_video_filepath)
                    print(f'Copied {original_video_path} to {new_video_filepath}')
                    calib_part_count += 1
                else:
                    print(f'{new_video_filepath} already exists')
            else:
                print(f'{original_video_path} does not exist')
        print(f'Copied {calib_part_count} calibration videos for participant {participant_id}')
        directory_copy_count += calib_part_count
    print(f'Copied {directory_copy_count} calibration videos')


if __name__ == '__main__':
    project_dir = '/media/iamshri/Seagate/QUB-PHEOVision'
    original_project_path = '/home/iamshri/Documents/PHEO-Data/Unprocessed'
    start_participant = 2
    end_participant = 5
    copy_calibration_videos(project_dir, start_participant, end_participant, original_project_path)
