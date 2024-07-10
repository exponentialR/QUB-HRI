import os
from pathlib import Path
import shutil

CAM_VIEWS = ['CAM_LL']


def check_exists(video_path):
    if not os.path.exists(video_path):
        print(f'{video_path} Does not exist!')
    else:
        pass


def transfer_calib_main(calib_vid_direc, destination_direc, vide_ext, vid_name, start_participant=1,
                        end_participant=70):
    """
    Directory format: pxx_cam_lx_task_xx.mp4
    Transfer files to respective directory
    Destination Directory if of the format

    pxx-----
            |____Cam_view
                        |______Task_xx.mp4

    :param calib_vid_direc:
    :param destination_direc:
    :param vide_ext:
    :param start_participant:
    :param end_participant:
    :return:
    """
    total_count = 0
    for participant_id in range(start_participant, end_participant):
        participant_id = f'p{participant_id:02d}'
        cam_copy_count = 0
        for cam_view in CAM_VIEWS:
            original_video_path = os.path.join(calib_vid_direc, f'{participant_id}_{cam_view}_{vid_name}.{vide_ext}')
            if os.path.exists(original_video_path):
                copy_path = os.path.join(destination_direc, participant_id, cam_view, f'{vid_name}.{vide_ext}')
                if os.path.exists(copy_path):
                    print(f'{copy_path} Already exist!')
                else:
                    shutil.copy(original_video_path, copy_path)
                    print(f'{original_video_path} copied to {copy_path}')
            else:
                print(f'{original_video_path} does not exist, please check!')
                continue


if __name__ == '__main__':
    PROJECT_DIR = '/media/BlueHDD/QUB-PHEO-datasets/LL_Original_Calib'
    VID_EXT = ('MP4')
    DESTINATION_DIREC = '/media/BlueHDD/Raw-QUB-PHEO/corrected'
    vid_name_copy = 'Original_CALIBRATION'
    transfer_calib_main(PROJECT_DIR, DESTINATION_DIREC, VID_EXT, vid_name_copy, start_participant=16,
                        end_participant=70)
