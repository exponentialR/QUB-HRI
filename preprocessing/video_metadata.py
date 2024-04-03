import os
import cv2
import csv


class VideoMetaDataExtractor:
    def __init__(self, proj_dir, output_csv, start_participant, end_participant):
        self.proj_dir = proj_dir
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.camera_views = ['CAM_LR', 'CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV']
        self.output_csv = output_csv
        self.video_metadata = []
