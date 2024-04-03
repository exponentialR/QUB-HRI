import cv2
import mediapipe as mp


class Visualizer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils

    def visualize(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (960, 540))  # Adjust based on your needs

            # Convert the BGR frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe solutions
            face_results = self.face_mesh.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)
            hand_results = self.hands.process(frame_rgb)

            # Draw the annotations on the frame
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    self.drawing_utils.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)

            if pose_results.pose_landmarks:
                self.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Display the resulting frame
            cv2.imshow('MediaPipe Results', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()


# Usage example


# Usage
if __name__ == "__main__":
    # video_path = "/home/iamshri/Documents/Dataset/Test_Evironment/p03/CAM_LR/BIAH_RB.mp4"  # Replace with the path to your video file
    video_path = '/home/iamshri/PycharmProjects/QUB-HRI/preprocessing/data/output/output.mp4'
    visualizer = Visualizer(video_path)
    visualizer.visualize()
