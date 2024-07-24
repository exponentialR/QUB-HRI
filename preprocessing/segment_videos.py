import os
import re
import json
import shutil
import subprocess
from tqdm import tqdm

CAM_VIEWS = ['CAM_AV', 'CAM_LL', 'CAM_LR', 'CAM_UR', 'CAM_UL']
TASKS = (
    'BIAH_BS', 'BIAH_BV', 'STAIRWAY_AP', 'STAIRWAY_MS', 'TOWER_HO', 'TOWER_MS', 'BRIDGE_AS', 'BRIDGE_BV', 'BIAH_RB')


def extract_bracket_letters(text):
    match = re.search(r'\((.*?)\)', text)
    return match.group(1) if match else None


def open_json_file(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def segment_videos(json_files_directory, base_videos_directory, output_directory):
    json_list = [file for file in os.listdir(json_files_directory) if file.endswith('.json') and file.startswith(TASKS)]
    print('LENGTH OF JSON LIST: ', len(json_list))
    skipped_files = []
    skipped_view = {view: 0 for view in CAM_VIEWS}
    for filename in tqdm(json_list, desc='Processing JSON files'):
        json_path = os.path.join(json_files_directory, filename)
        data = open_json_file(json_path)
        for task in tqdm(data, desc='Segmenting videos'):
            video_url = task['video_url']
            for cam_view in CAM_VIEWS:
                vid_url = video_url.split('-')[-1].replace('CAM_AV_P', cam_view).split('_')
                for trick in tqdm(task['tricks'], desc=f'Processing tricks'):
                    label_abv = extract_bracket_letters(trick['labels'][0])
                    label_dir = os.path.join(output_directory, label_abv)
                    # os.makedirs(label_dir, exist_ok=True)  # Ensure the directory exists

                    start_time = str(trick['start'])
                    end_time = str(trick['end'])
                    output_filename = f'{vid_url[0]}-{cam_view}-{vid_url[3]}_{vid_url[-1][0:-4]}-{label_abv}-{start_time}_{end_time}.mp4'
                    output_path = os.path.join(label_dir, output_filename)
                    if os.path.exists(output_path) and os.path.getsize(output_path) < 100000:
                        # shutil.move(output_path, os.path.join(corrupted_seg_dir, output_filename))
                        print(f'Moved corrupted file {output_path} to {corrupted_seg_dir}')
                        skipped_view[cam_view] += 1
                        skipped_files.append(output_path)


                    # vid_path = os.path.join(base_videos_directory, vid_url[0],
                    #                         f'{vid_url[1]}_{vid_url[2]}/{vid_url[3]}_{vid_url[4]}')

                    # if os.path.exists(vid_path):
                    #     command = [
                    #         'ffmpeg', '-i', vid_path, '-ss', start_time, '-to', end_time, '-c', 'copy',
                    #         '-n',
                    #         output_path
                    #     ]
                    #     # subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    #     print(f'Segmented video saved at {output_path}')
                    # else:
                    #     continue
    print('Skipped files: ', skipped_files)
    print('Total skipped files: ', len(skipped_files))
    print('Skipped views: ', skipped_view)


if __name__ == '__main__':
    json_files_directory = '/home/samueladebayo/PycharmProjects/QUB-HRI/actionLabelling/labels'
    base_videos_directory = '/media/samueladebayo/EXTERNAL_USB/QUB-PHEO-Proceesed'
    output_directory = '/home/samueladebayo/Documents/PhD/QUB-PHEO-Dataset/Annotated'
    os.makedirs(output_directory, exist_ok=True)
    corrupted_seg_dir = '/home/samueladebayo/Documents/PhD/QUB-PHEO-Dataset/corrupted-segment'
    segment_videos(json_files_directory, base_videos_directory, output_directory)

