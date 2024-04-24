import os
import zipfile
from tqdm import tqdm

import os
import zipfile
from tqdm import tqdm


def unzip_to_folder(zip_path, extract_dir):
    # Ensure the extraction directory exists
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(path=extract_dir)

    # Loop over the directories in the extraction directory
    for root, dirs, files in os.walk(extract_dir):
        for filename in files:
            if filename.endswith('.zip'):
                # Construct the full path to the file
                file_path = os.path.join(root, filename)
                # Define the directory to extract its contents
                extract_path = os.path.join(root, os.path.splitext(filename)[0])
                # Create the directory if it doesn't exist
                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)
                # Unzip the file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract all the contents into the directory with a tqdm progress bar
                    for member in tqdm(zip_ref.infolist(), desc=f"Unzipping {filename}", leave=False):
                        zip_ref.extract(member, extract_path)
                # Remove the zip file after extraction
                os.remove(file_path)

    print(f"Unzipping complete. Files extracted to: {extract_dir}")


if __name__ == '__main__':
    # Usage example
    zip_file_path = '/home/qub-hri/Documents/Datasets/QUB-PHEO/p01-p02.zip'
    start_participant, end_participant = 1, 2
    destination_dir = os.path.join(os.path.dirname(zip_file_path), f'p{start_participant:02d}-p{end_participant:02d}')
    unzip_to_folder(zip_file_path, destination_dir)
