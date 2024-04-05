import random
import os
import json

random.seed(42)


# FILE_PATH = 'task_random.json'


def task_random_generate(start_participant=8, end_participant=100, file_path='tasks.json'):
    tasks = {
        'BIAH': ['BS', 'RB', 'BV'],
        'BRIDGE': ['AS', 'BV'],
        'CALIBRATION': [],
        'STAIRWAY': ['AP', 'MS'],
        'TOWER': ['MS', 'HO']
    }

    task_random_list = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            task_random_list = json.load(file)
    else:
        with open(file_path, 'w') as file:
            json.dump(task_random_list, file)

    for part_id in range(start_participant, end_participant + 1):
        part_id_str = f'p{part_id-1:02d}'
        task_keys = [key for key in tasks.keys() if key != 'CALIBRATION']
        random.shuffle(task_keys)
        task_keys.insert(0, 'CALIBRATION')

        expanded_task_list = []
        for task in task_keys:
            if tasks[task]:
                for suffix in tasks[task]:
                    expanded_task_list.append(f'{task}_{suffix}')
            else:
                expanded_task_list.append(task)

        if part_id_str not in task_random_list:
            task_random_list[part_id_str] = expanded_task_list

    with open(file_path, 'w') as f:
        json.dump(task_random_list, f, indent=4)
    print(f'Metadata successfully updated in {file_path}')
    return file_path


if __name__ == '__main__':
    task_random_generate()
