import os
import re
import json
import subprocess
from tqdm import tqdm

CAM_VIEWS = ['CAM_AV', 'CAM_LL', 'CAM_LR', 'CAM_UR', 'CAM_UL']
TASKS = (
    'BIAH_BS', 'BIAH_BV', 'STAIRWAY_AP', 'STAIRWAY_MS', 'TOWER_HO', 'TOWER_MS', 'BRIDGE_AS', 'BRIDGE_BV', 'BIAH_RB')


def extract_bracket_letters(text):
    # Use regex to find the text within brackets
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)
    else:
        return None


def open_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data


def segment_videos(json_files_directory, base_videos_directory, output_directory):
    """
    Segment the videos based on the json files
    """
    json_list = [file for file in os.listdir(json_files_directory) if file.endswith('.json') and file.startswith(TASKS)]
    print('LENGTH OF JSON LIST: ', len(json_list))

    # Wrap the outer loop with tqdm for a progress bar
    for filename in tqdm(json_list, desc='Processing JSON files'):
        json_path = os.path.join(json_files_directory, filename)
        data = open_json_file(json_path)
        for task in tqdm(data, desc='Segmenting videos'):
            video_url = task['video_url']
            for cam_view in CAM_VIEWS:
                vid_url = video_url.split('-')[-1]
                vid_url = vid_url.replace('CAM_AV_P', cam_view)
                vid_url = vid_url.split('_')
                segmented_path = os.path.join(output_directory,
                                              f'{vid_url[0]}_{vid_url[1]}_{vid_url[2]}_{vid_url[3]}_{vid_url[4][:-4]}')

                vid_path = os.path.join(base_videos_directory, vid_url[0],
                                        f'{vid_url[1]}_{vid_url[2]}/{vid_url[3]}_{vid_url[4]}')
                if os.path.exists(vid_path):
                    for trick in tqdm(task['tricks'], desc=f'Cutting videos at {vid_path}'):
                        start_time = str(trick['start'])
                        end_time = str(trick['end'])
                        label_abv = extract_bracket_letters(trick['labels'][0])
                        output_filename = f'{segmented_path}_{label_abv}_{start_time}_{end_time}.mp4'

                        command = [
                            'ffmpeg', '-i', vid_path, '-ss', start_time, '-to', end_time, '-c', 'copy',
                            '-y',
                            output_filename
                        ]
                        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress output
                else:
                    print(f'Video {vid_path} does not exist. Skipping...')


if __name__ == '__main__':
    json_files_directory = '/home/iamshri/PycharmProjects/QUB-HRI/actionLabelling/labels'
    base_videos_directory = '/home/iamshri/ml_projects/Datasets/test-qub-pheo'
    output_directory = '/home/iamshri/ml_projects/Datasets/qub-pheo-segmented-Videos'
    os.makedirs(output_directory, exist_ok=True)

    segment_videos(json_files_directory, base_videos_directory, output_directory)
