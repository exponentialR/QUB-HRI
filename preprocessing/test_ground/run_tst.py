import subprocess

from moviepy.editor import VideoFileClip, AudioFileClip

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('MoviePy').setLevel(logging.CRITICAL)
logging.getLogger('moviepy').setLevel(logging.CRITICAL)


def get_video_bitrate(video_path):
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


def add_audio_var_video(video_path: str = None, output_video_path: str = None, long_audio_clip: AudioFileClip = None):
    original_bitrate = get_video_bitrate(video_path)
    video_clip = VideoFileClip(video_path)
    trimmed_audio_clip = long_audio_clip.subclip(0, video_clip.duration)
    final_video_clip = video_clip.set_audio(trimmed_audio_clip)
    final_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', bitrate=original_bitrate,
                                     verbose=False)
    return output_video_path


if __name__ == '__main__':
    # video_path = '/home/qub-hri/Documents/Datasets/QUB-PHEO/p61/CAM_AV/STAIRWAY_AP.mp4'
    video_path = '/home/qub-hri/Documents/Datasets/QUB-PHEO/p61/CAM_AV/BIAH_BV.mp4'
    audio_path = '../audio.mp3'
    output_video_path = 'output_video.mp4'
    print(get_video_bitrate(video_path))
    long_audio_clip = AudioFileClip(audio_path)
    add_audio_var_video(video_path, output_video_path, long_audio_clip)
