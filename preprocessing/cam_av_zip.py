import os
import zipfile


def zipdir(proj_dir, start_participant, end_participant):
    # paths for each participant's directory
    participant_directories = [
        os.path.join(proj_dir, f'p{part_id:02d}', 'CAM_AV_P')
        for part_id in range(start_participant, end_participant + 1)
    ]

    # Loop through each participant directory
    for participant_dir in participant_directories:
        # Create a ZIP file name based on the directory name
        zip_filename = f"{participant_dir}.zip"

        # Create a ZipFile object in WRITE mode
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the directory structure
            for root, dirs, files in os.walk(participant_dir):
                for file in files:
                    # Create the complete filepath of the file in directory
                    file_path = os.path.join(root, file)
                    # Add file to zip archive
                    # arcname handles the relative path of the file inside the archive
                    arcname = os.path.relpath(file_path, start=participant_dir)
                    zipf.write(file_path, arcname)
        print('Zipped:', participant_dir)

    print("Zipping complete for all directories.")


if __name__ == '__main__':
    proj_dir = ''
    start_participant = 1
    end_participant = 10
    zipdir(proj_dir, start_participant, end_participant)
