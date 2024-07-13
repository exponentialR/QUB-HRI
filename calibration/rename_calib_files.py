import os
import shutil

if __name__ == '__main__':
    START_PARTICIPANT = 17
    END_PARTICIPANT = 70

    CAM_VIEWS = ['CAM_AV', 'CAM_UL', 'CAM_UR', 'CAM_LR', 'CAM_LL']
    PROJECT_DIR = '/media/samueladebayo/EXTERNAL_USB/QUB-PHEO-Proceesed'
    total_calib_count = 0
    for participant_id in range(START_PARTICIPANT, END_PARTICIPANT + 1):
        participant_id = f'p{participant_id:02d}'
        for cam_view in CAM_VIEWS:

            current_dir = os.path.join(PROJECT_DIR, participant_id, cam_view)
            if not os.path.exists(current_dir):
                continue
            list_files = [file for file in os.listdir(current_dir) if
                          file.lower().endswith('npz') and (file.lower().startswith('intrinsic') or file.lower().startswith('calib_param')) ]
            for file in list_files:
                current_calib = os.path.join(current_dir, file)
                # print(current_calib)
                new_name = os.path.join(current_dir, f'{participant_id}-{cam_view}-intrinsic.npz')
                # print(current_calib)
                if os.path.exists(current_calib):
                    total_calib_count += 1

                    # print(f'Calibration {current_calib} file already exist!')
                    os.rename(current_calib, new_name)
                    print(f'{current_calib} file renamed as {new_name}')

                else:
                    continue
                    # shutil.copy(current_calib, new_name)
    print(f'Total Calibration Count is {total_calib_count}')