import os
import shutil
import zipfile
from tqdm import tqdm

def zipdir(proj_dir, start_participant, end_participant, output_dir):
    # Create a temporary directory for participant zips
    temp_dir = os.path.join(output_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Generate paths for each participant's directory
    participant_directories = [
        os.path.join(proj_dir, f'p{part_id:02d}', 'CAM_AV_P')
        for part_id in range(start_participant, end_participant + 1)
    ]

    # Loop through each participant directory with a tqdm progress bar
    for participant_dir in tqdm(participant_directories, desc="Zipping Participants"):
        # Get participant identifier, e.g., p01, p02
        participant_id = os.path.basename(os.path.dirname(participant_dir))

        # Create specific temporary directory for each participant
        specific_temp_dir = os.path.join(temp_dir, participant_id)
        if not os.path.exists(specific_temp_dir):
            os.makedirs(specific_temp_dir)

        # Define the zip file name based on the directory name and temporary directory
        zip_filename = os.path.join(specific_temp_dir, f"{os.path.basename(participant_dir)}.zip")

        # Check if the zip file already exists before creating it
        if not os.path.exists(zip_filename):
            # Create a ZipFile object in WRITE mode
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Get all files to be zipped for tqdm progress bar
                file_paths = []
                for root, dirs, files in os.walk(participant_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_paths.append((file_path, os.path.relpath(file_path, start=participant_dir)))

                # Add files to zip archive with progress bar
                for file_path, arcname in tqdm(file_paths, desc=f"Zipping {participant_id}", leave=False):
                    zipf.write(file_path, arcname)
            print(f'Zipped Participant {participant_id}/CAM_AV_P')
        else:
            print(f'Skipped zipping {participant_id}/CAM_AV_P as zip file already exists.')

    # Check if the final output zip file already exists
    final_zip_path = os.path.join(output_dir, f'p{start_participant:02d}-p{end_participant:02d}.zip')
    if not os.path.exists(final_zip_path):
        # Zip the temporary directory after all participant zips are done
        with zipfile.ZipFile(final_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=temp_dir)
                    zipf.write(file_path, arcname)
        print("Final zipping complete. All participant directories are zipped into:", final_zip_path)
    else:
        print("Skipped final zipping as 'final_output.zip' already exists.")
    
    # Clean up: remove the temporary directory
    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    proj_dir = '/media/BlueHDD/QUB-PHEO-datasets'
    start_participant = 1
    end_participant = 5
    output_dir = '/media/BlueHDD/zipped_folder'
    zipdir(proj_dir, start_participant, end_participant, output_dir=output_dir)
