import os
import shutil
from tqdm import tqdm


def copy_files(proj_dir, view, output_dir):
    list_subtask = sorted([sub for sub in os.listdir(proj_dir) if os.path.isdir(os.path.join(proj_dir, sub))])
    print(len(list_subtask))
    output_dir = os.path.join(output_dir, view)
    os.makedirs(output_dir, exist_ok=True)
    file_count = 0

    for subtask in tqdm(list_subtask, desc="Copying subtasks"):
        subtask_dir = os.path.join(proj_dir, subtask)
        output_land_dir = os.path.join(output_dir, subtask)
        os.makedirs(output_land_dir, exist_ok=True)
        print(output_land_dir)

        files = [f for f in os.listdir(subtask_dir) if f.endswith('.h5') and f.split('.h5')[0].split('-')[1] == view]
        for file in files:
            original_file = os.path.join(subtask_dir, file)
            output_file = os.path.join(output_land_dir, file)
            shutil.copyfile(original_file, output_file)
            file_count += 1

    print(f"Total files copied: {file_count} FOR VIEW: {view}")


if __name__ == '__main__':
    proj_dir = '/home/samueladebayo/Documents/PhD/QUBPHEO/landmark'
    output_dir = '/home/samueladebayo/Documents/PhD/QUBPHEO/LANDMARK'
    view = 'CAM_LR'
    copy_files(proj_dir, view, output_dir)
