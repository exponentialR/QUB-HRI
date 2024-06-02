import os
import zipfile
import sys
from tqdm import tqdm


def zip_folders_in_groups(directory, start_participant, end_participant, group_size=3):
    # Get the list of folders in the directory
    folders = sorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])

    # Filter folders based on start and end participants
    folders = [f for f in folders if int(f[1:]) >= start_participant and int(f[1:]) <= end_participant]

    # Create zips in groups of specified size
    for i in tqdm(range(0, len(folders), group_size), desc="Overall Progress", unit="group"):
        group_folders = folders[i:i + group_size]
        if group_folders:
            zip_filename = f"{group_folders[0]}_to_{group_folders[-1]}.zip"
            zip_filepath = os.path.join(directory, zip_filename)

            with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                for folder in tqdm(group_folders, desc=f"Zipping {zip_filename}", leave=False):
                    folder_path = os.path.join(directory, folder)
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, start=directory)
                            zipf.write(file_path, arcname)
            print(f"Created {zip_filename}")


if __name__ == "__main__":
    directory = sys.argv[1]
    start_participant = int(sys.argv[2])
    end_participant = int(sys.argv[3])
    zip_folders_in_groups(directory, start_participant, end_participant)
