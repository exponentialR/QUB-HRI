import os

folders = ['calibration_videos', 'videos_to_correct', 'corrected_videos']

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
