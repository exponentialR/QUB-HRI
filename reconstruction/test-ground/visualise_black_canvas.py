import cv2
import mediapipe as mp
import numpy as np


class Visualizer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.3,
                                           min_tracking_confidence=0.3)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2,
                                              min_tracking_confidence=0.2)
        self.drawing_utils = mp.solutions.drawing_utils

        # Custom drawing specifications for thicker landmarks and connections
        self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=4,
                                                                            color=(0, 255, 0))
        self.connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=7, color=(0, 0, 255))

    def visualize(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (960, 540))  # Adjust based on your needs

            # Create a black frame
            black_frame = np.zeros(frame.shape, dtype=np.uint8)

            # Convert the BGR frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe solutions
            face_results = self.face_mesh.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)
            hand_results = self.hands.process(frame_rgb)

            # Draw the annotations on the black frame with custom specifications
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    self.drawing_utils.draw_landmarks(
                        black_frame,
                        face_landmarks,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.landmark_drawing_spec,
                        connection_drawing_spec=self.connection_drawing_spec)

            if pose_results.pose_landmarks:
                self.drawing_utils.draw_landmarks(
                    black_frame,
                    pose_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec,
                    connection_drawing_spec=self.connection_drawing_spec)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.drawing_utils.draw_landmarks(
                        black_frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_drawing_spec,
                        connection_drawing_spec=self.connection_drawing_spec)

            # Display the resulting frame (black background with landmarks)
            cv2.imshow('MediaPipe Results', black_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()


# Usage example
# video_path = "/home/iamshri/Documents/Dataset/Test_Evironment/p03/CAM_LL/BIAH_RB.mp4"  # Replace with the path to your video file
video_path = '/home/iamshri/PycharmProjects/QUB-HRI/preprocessing/data/output/output.mp4'
visualizer = Visualizer(video_path)
visualizer.visualize()
