import os
import cv2
import csv


class VideoMetadataExtractor:
    def __init__(self, proj_dir, output_csv, start_participant, end_participant):
        self.proj_dir = proj_dir
        self.output_csv = output_csv
        self.start_participant = start_participant
        self.end_participant = end_participant
        self.camera_views = ['CAM_LR', 'CAM_LL', 'CAM_UR', 'CAM_UL', 'CAM_AV']
        self.existing_data = self.load_existing_data()

    def load_existing_data(self):
        data = set()
        if os.path.exists(self.output_csv) and os.stat(self.output_csv).st_size > 0:
            with open(self.output_csv, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for row in reader:
                    data.add(tuple(row))
        return data

    def extract_and_save(self):
        write_header = not os.path.exists(self.output_csv) or os.stat(self.output_csv).st_size == 0
        with open(self.output_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['Participant', 'Camera View', 'Video File', 'Duration (s)', 'FPS', 'Frame Count'])

            for i in range(self.start_participant, self.end_participant + 1):
                participant_dir = os.path.join(self.proj_dir, f'p{i:02d}')
                for cam_view in self.camera_views:
                    cam_dir = os.path.join(participant_dir, cam_view)
                    if not os.path.exists(cam_dir):
                        continue
                    for vid_file in os.listdir(cam_dir):
                        if vid_file.endswith('.mp4'):
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
    proj_dir = "/media/qub-hri/EXTERNAL_USB/QUB-PHEO"
    output_csv = "/media/qub-hri/EXTERNAL_USB/QUB-PHEO/video_metadata.csv"
    start_participant = 1
    end_participant = 50

    metadata_extractor = VideoMetadataExtractor(proj_dir, output_csv, start_participant, end_participant)
    metadata_extractor.extract_and_save()
