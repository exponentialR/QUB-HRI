import cv2
import mediapipe as mp
from gaze_aerial import compute_aerial_gaze
# from ultralytics import yolov5
import h5py
import numpy as np
from tqdm import tqdm


class LandmarksToHDF5:
    def __init__(self, vid_path, landmark_hdf5_path, detect_confidence=0.5, track_confidence=0.5, logger=None):
        self.video_path = vid_path
        self.hdf5_path = landmark_hdf5_path
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=detect_confidence,
                                                         min_tracking_confidence=track_confidence)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=detect_confidence,
                                           min_tracking_confidence=track_confidence)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                              min_detection_confidence=detect_confidence,
                                              min_tracking_confidence=track_confidence)
        self.logger = logger

    def process_and_save(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with h5py.File(self.hdf5_path, 'w') as f:
            face_dset = f.create_dataset("face_landmarks", (frame_count, 478, 3), dtype='f')
            pose_dset = f.create_dataset("pose_landmarks", (frame_count, 33, 4), dtype='f')
            hands_dset = f.create_dataset("hand_landmarks", (frame_count, 2, 21, 3),
                                          dtype='f')

            with tqdm(total=frame_count, desc=f"Processing Video {self.video_path}") as pbar:
                frame_idx = 0
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = self.face_mesh.process(frame_rgb)
                    pose_results = self.pose.process(frame_rgb)
                    hand_results = self.hands.process(frame_rgb)

                    # Store face landmarks
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            for idx, landmark in enumerate(face_landmarks.landmark):
                                face_dset[frame_idx, idx] = (landmark.x, landmark.y, landmark.z)
                    else:
                        face_dset[frame_idx] = np.zeros((478, 3))

                    # Store pose landmarks
                    if pose_results.pose_landmarks:
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            pose_dset[frame_idx, idx] = (landmark.x, landmark.y, landmark.z, landmark.visibility)
                    else:
                        pose_dset[frame_idx] = np.zeros((33, 4))

                    # Store hand landmarks
                    hands_data = np.zeros((2, 21, 3))  # Reset hands data for each frame
                    if hand_results.multi_hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                            for idx, landmark in enumerate(hand_landmarks.landmark):
                                hands_data[hand_idx, idx] = (landmark.x, landmark.y, landmark.z)
                    hands_dset[frame_idx] = hands_data

                    frame_idx += 1
                    pbar.update(1)

        cap.release()
        print(f"Landmarks saved to HDF5 : {self.hdf5_path}")


if __name__ == '__main__':
    video_path = "/home/iamshri/Documents/Dataset/Test_Evironment/p03/CAM_LL/BIAH_RB.mp4"  # Replace with the path to your video file
    hdf5_path = 'landmarks_data.hdf5'  # Path where you want to save the HDF5 file
    processor = LandmarksToHDF5(video_path, hdf5_path)
    processor.process_and_save()
