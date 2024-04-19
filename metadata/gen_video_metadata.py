import os
import cv2
import csv


class VideoMetadataExtractor:
    def __init__(self, proj_dir, output_csv, start_participant, end_participant, with_calib=False):
        self.proj_dir = proj_dir
        self.output_csv = output_csv
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.with_calib = with_calib
        self.camera_views = ['CAM_LR', 'CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV']
        self.ensure_csv_exists()
        self.existing_data = self.load_existing_data()
    
    def ensure_csv_exists(self):
        if not os.path.exists(self.output_csv) or os.stat(self.output_csv).st_size == 0:
            with open(self.output_csv, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Participant', 'Camera View', 'Video File', 'Duration (s)', 'FPS', 'Frame Count'])


    
    def load_existing_data(self):
        data = set()
        with open(self.output_csv, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                data.add(tuple(row))
        return data

    def extract_and_save(self):
        with open(self.output_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            for i in range(self.start_participant, self.end_participant + 1):
                participant_dir = os.path.join(self.proj_dir, f'p{i:02d}')
                for cam_view in self.camera_views:
                    cam_dir = os.path.join(participant_dir, cam_view)
                    if not os.path.exists(cam_dir):
                        continue
                    for vid_file in os.listdir(cam_dir):
                        if vid_file.endswith('.mp4'):
                            if not self.with_calib and vid_file.lower().startswith('calib'):
                                continue
                            vid_path = os.path.join(cam_dir, vid_file)
                            duration, fps, frame_count = self.get_video_metadata(vid_path)
                            new_row = (f'p{i:02d}', cam_view, vid_file, str(duration), str(fps), str(frame_count))
                            if new_row not in self.existing_data:
                                writer.writerow(new_row)
                                self.existing_data.add(new_row)

    def get_video_metadata(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return None, None, None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        cap.release()
        return duration, fps, frame_count

if __name__ == '__main__':
    proj_dir = "/media/BlueHDD/QUB-PHEO-datasets"
    if not os.path.exists(proj_dir):
        print('Directory does not exist')
    else:
        pass
    output_csv = "/media/BlueHDD/QUB-PHEO-datasets/qub-pheo_metadata_withcalib.csv"
    start_participant = 1
    end_participant = 64

    metadata_extractor = VideoMetadataExtractor(proj_dir, output_csv, start_participant, end_participant, with_calib=True)
    metadata_extractor.extract_and_save()