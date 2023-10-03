from goprocam import GoProCamera, constants
from datetime import datetime
import os


def get_filename(participant_id, task_name, version_number, camera_position, data_type):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"{task_name}_{version_number}_{camera_position}_{data_type}_{timestamp}.mp4"
    return filename


def main():
    participant_id = input("Enter participant ID: ")
    task_name = input("Enter task name: ")
    version_number = input("Enter version number: ")

    # Create participant directory if it doesn't exist
    participant_dir = f"ProjectRoot/Participant_{participant_id}"
    os.makedirs(participant_dir, exist_ok=True)

    camera_positions = ['FrontLeft', 'FrontRight', 'SideLeft', 'SideRight', 'Aerial']

    for camera_position in camera_positions:
        filename = get_filename(participant_id, task_name, version_number, camera_position, 'Video')
        file_path = os.path.join(participant_dir, filename)

        # Assuming each camera_position corresponds to a unique GoPro SSID
        gopro = GoProCamera.GoPro(ssid=f"GoPro{camera_position}")
        gopro.shutter(constants.start)  # Start recording
        print(f"Recording started: {file_path}")

        # Wait for user input to stop recording, or implement your logic
        input("Press Enter to stop recording...")

        gopro.shutter(constants.stop)  # Stop recording
        print(f"Recording stopped: {file_path}")

        # Download the last recorded video from the GoPro
        media = gopro.downloadLastMedia(custom_filename=file_path)
        print(f"Video saved to: {media}")


if __name__ == "__main__":
    main()
