import cv2
import os
import json
import pandas as pd


class CompareFrameCount:
    def __init__(self, json_file_path, output_csv_path, same_frames_csv_path, proj_dir):
        self.json_file_path = json_file_path
        self.output_csv_path = output_csv_path
        self.same_frames_csv_path = same_frames_csv_path
        self.proj_dir = proj_dir  # Base directory for all video files
        self.data = self.load_json()
        self.frame_count_discrepancies = []
        self.same_frame_counts = []

    def load_json(self):
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def extract_details(self, filename):
        parts = filename.split('-')
        participant = parts[0]
        camera_view = parts[1]
        task = parts[2]
        subtask = parts[3]
        timestamp = parts[4].replace('.mp4', '')
        return participant, camera_view, task, subtask, timestamp

    def count_frames(self, filepath):
        cap = cv2.VideoCapture(filepath)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frames

    def compare_frame_count(self):
        grouped_files = {}
        for path in self.data:
            filename = os.path.basename(path)
            participant, camera_view, task, subtask, timestamp = self.extract_details(filename)
            key = (participant, task, subtask, timestamp)
            if key not in grouped_files:
                grouped_files[key] = []
            subtask_path = os.path.join(self.proj_dir, subtask)  # Append subtask directory
            full_path = os.path.join(subtask_path, filename)  # Complete path with filename
            grouped_files[key].append((camera_view, full_path))

        for key, views in grouped_files.items():
            if len(views) > 1:
                frame_counts = {view: self.count_frames(path) for view, path in views}
                frame_values = list(frame_counts.values())
                if len(set(frame_values)) > 1:  # Check if all frame counts are the same
                    self.frame_count_discrepancies.append({
                        "participant": key[0],
                        "task": key[1],
                        "subtask": key[2],
                        "timestamp": key[3],
                        "details": frame_counts
                    })
                else:
                    for view, path in views:
                        self.same_frame_counts.append({
                            "video_path": path,
                            "frame_count": frame_values[0]
                        })

    def save_discrepancies_to_csv(self):
        df = pd.DataFrame(self.frame_count_discrepancies)
        df.to_csv(self.output_csv_path, index=False)
        print(f"Discrepancies saved to {self.output_csv_path}")

    def save_same_frame_counts_to_csv(self):
        df = pd.DataFrame(self.same_frame_counts)
        df.to_csv(self.same_frames_csv_path, index=False)
        print(f"Same frame counts details saved to {self.same_frames_csv_path}")

    def run(self):
        self.compare_frame_count()
        self.save_discrepancies_to_csv()
        self.save_same_frame_counts_to_csv()


if __name__ == "__main__":
    proj_dir = '/home/iamshri/ml_projects/Datasets/QUB-PHEO-segmented-Videos'
    json_file_path = '/home/iamshri/ml_projects/Datasets/QUB-PHEO-segmented-Videos/subtask_labellist.json'
    output_csv_path = '/home/iamshri/ml_projects/Datasets/QUB-PHEO-segmented-Videos/frame_count_discrepancies.csv'
    same_frames_csv_path = '/home/iamshri/ml_projects/Datasets/QUB-PHEO-segmented-Videos/same_frame_counts.csv'
    comparer = CompareFrameCount(json_file_path, output_csv_path, same_frames_csv_path, proj_dir)

    comparer.run()
