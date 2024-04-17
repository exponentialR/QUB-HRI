import os.path
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip
from logger_utils import setup_calibration_video_logger
import logging


class AddAudioData:
    """
    This adds audio to the videos in the Camera Aerial Perspective
    """

    def __init__(self, project_dir: str, start_participant: int = 1, end_participant: int = 10,
                 audio_path: str = 'audio.mp3'):
        self.project_dir = project_dir
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.participant_dir = [os.path.join(self.project_dir, f'p{participant_id:02d}') for participant_id in
                                range(self.start_participant, self.end_participant + 1)]
        self.long_audio_clip = AudioFileClip(audio_path)
        self.logger = setup_calibration_video_logger(
            "Video-Synchronization-Logger",
            format_str='%(asctime)s - %(name)s - [Task: %(task_name)s] - [Detail: %(detail)s] - %(levelname)s - %('
                       'message)s',
            extra_attrs=['task_name', 'detail'],
            error_log_file='vid_audio-logs.txt',
            levels_to_save={logging.DEBUG, logging.INFO},  # Set levels to save
            console_level=logging.INFO  # Set console level
        )
        self.logger.info(f' Current Project Directory is {self.project_dir}')
        self.logger.info(
            f' Will now add audio to all videos in p{self.start_participant:02d}/CAM_AV to p{self.end_participant:02d}/CAM_AV')
        self.logger.info(
            f' Videos with added audio will be saved in p{self.start_participant:02d}/CAM_AV_P to p{self.end_participant:02d}/CAM_AV_P')

    def add_max_audio(self):
        """
        This adds audio to every videos in the camera aerial in participants directory,
         where the audio length is affixed based on the length of the video.
        :return:
        """
        total_video_count = 0
        for participant_dir in self.participant_dir:

            current_camera_directory = os.path.join(participant_dir, 'CAM_AV')
            new_AV_directory = os.path.join(participant_dir, 'CAM_AV_P')
            if not os.path.exists(new_AV_directory):
                os.makedirs(new_AV_directory, exist_ok=True)
            else:
                continue

            part_video_count = 0
            for video_file in os.listdir(current_camera_directory):
                if video_file.endswith(('mp4', 'avi')):
                    current_video = os.path.join(current_camera_directory, video_file)

                    new_video_path = os.path.join(new_AV_directory, video_file)
                    current_participant = os.path.basename(participant_dir)

                    output_video_path = self.add_audio_var_video(current_video, new_video_path,
                                                                 long_audio_clip=self.long_audio_clip)
                    self.logger.debug(f'Audio added to {current_participant}/CAM_AV/{video_file}')
                    print(f'Output Video found in {output_video_path}')
                    part_video_count += 1
            print(f'Audio added to {part_video_count} videos in {self.participant_dir}')
            total_video_count += part_video_count
            self.logger.info(f'Audio added to {total_video_count} videos')

    def get_video_bitrate(self, video_path):
        # Use ffprobe to get the bitrate of the video
        try:
            result = subprocess.check_output(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=bit_rate', '-of',
                 'default=noprint_wrappers=1:nokey=1', video_path])
            bitrate = result.strip().decode('utf-8')
            print(bitrate)
            return bitrate
        except subprocess.CalledProcessError as e:
            print(f"Failed to get bitrate for video {video_path}: {str(e)}")
            return None

    def add_audio_var_video(self, video_path: str = None, output_video_path: str = None,
                            long_audio_clip: AudioFileClip = None):
        original_bitrate = self.get_video_bitrate(video_path)
        video_clip = VideoFileClip(video_path)
        trimmed_audio_clip = long_audio_clip.subclip(0, video_clip.duration)
        final_video_clip = video_clip.set_audio(trimmed_audio_clip)
        final_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac',
                                         bitrate=original_bitrate, verbose=False)
        return output_video_path


if __name__ == '__main__':
    project_path = '/home/qub-hri/Documents/Datasets/QUB-PHEO'
    Project = AddAudioData(project_path, 61, 61)
    Project.add_max_audio()
