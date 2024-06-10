import cv2
import mediapipe as mp
import numpy as np


def draw_2d_grid(image, grid_size=40, line_color=(200, 200, 200), thickness=1):
    height, width, _ = image.shape
    # Draw vertical lines
    for x in range(0, width, width // grid_size):
        cv2.line(image, (x, 0), (x, height), line_color, thickness)
    # Draw horizontal lines
    for y in range(0, height, height // grid_size):
        cv2.line(image, (0, y), (width, y), line_color, thickness)


class Visualizer:
    def __init__(self, video_path, show_landmark=True):
        self.video_path = video_path
        self.show_landmark = show_landmark
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.3,
                                           min_tracking_confidence=0.3)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2,
                                              min_tracking_confidence=0.2)
        self.drawing_utils = mp.solutions.drawing_utils

        # Custom drawing specifications for thicker landmarks and connections
        self.face_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=4,
                                                                                 color=(0, 255, 0))
        self.pose_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=4,
                                                                                 color=(255, 0, 0))
        self.hand_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=4,
                                                                                 color=(0, 0, 255))
        self.pose_connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=4, color=(
            255, 0, 0))  # Thicker cyan lines for pose

    def visualize(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_number = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_number += 1

            # Resize frame for faster processing
            frame = cv2.resize(frame, (960, 540))

            # Create a white frame

            # text = f'Participant P03 | TASK: BIAH | Frame No: {frame_number:04d}'
            # font = cv2.FONT_HERSHEY_DUPLEX
            # font_scale = 0.5
            # thickness = 1
            # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            # text_x = (frame.shape[1] - text_size[0]) // 2
            # text_y = 80  # Set vertical position for text
            # cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 250), thickness)

            # Convert the BGR frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.show_landmark:
                text = f'Participant P03 | TASK: BIAH | Frame No: {frame_number:04d}'
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.45
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                # # text_x = (frame.shape[1] - text_size[0]) // 2
                # text_x = (frame.shape[1] - text_size[0])//2 - 200
                # text_y = 80  # Set vertical position for text
                # cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 250), thickness)

                white_frame = np.ones(frame.shape, dtype=np.uint8) * 255
                text_x = ((white_frame.shape[1] - text_size[0]) // 2) + 300
                text_y = 60 + text_size[1]
                # cv2.putText(white_frame, text, (text_x, text_y), font, font_scale, (100, 150, 0), thickness)
                cv2.putText(white_frame, text, (text_x, text_y), font, font_scale, (255, 0, 255), thickness)
                # Process with MediaPipe solutions
                face_results = self.face_mesh.process(frame_rgb)
                pose_results = self.pose.process(frame_rgb)
                hand_results = self.hands.process(frame_rgb)

                # Draw the annotations on the white frame with custom specifications
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        self.drawing_utils.draw_landmarks(
                            white_frame,
                            face_landmarks,
                            mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=self.face_landmark_drawing_spec,
                            connection_drawing_spec=self.pose_connection_drawing_spec)

                if pose_results.pose_landmarks:
                    self.drawing_utils.draw_landmarks(
                        white_frame,
                        pose_results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.pose_landmark_drawing_spec,
                        connection_drawing_spec=self.pose_connection_drawing_spec)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.drawing_utils.draw_landmarks(
                            white_frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.hand_landmark_drawing_spec,
                            connection_drawing_spec=self.pose_connection_drawing_spec)


                legend_start_y = 100
                text_offset_x = 20
                legend_text_position_x = white_frame.shape[1] - 200  # Adjust as needed
                # cv2.putText(white_frame, f'Participant P03 | TASK: Block in a Hole Frame {frame_number:03d}',(legend_text_position_x-200, legend_start_y) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

                cv2.circle(white_frame, (legend_text_position_x, legend_start_y), 10, (0, 255, 0), -1)  # Green dot
                cv2.putText(white_frame, 'Face Landmark', (legend_text_position_x + text_offset_x, legend_start_y + 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                cv2.circle(white_frame, (legend_text_position_x, legend_start_y + 30), 10, (255, 0, 0), -1)  # Red dot
                cv2.putText(white_frame, 'Pose Landmark', (legend_text_position_x + text_offset_x, legend_start_y + 35),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                cv2.circle(white_frame, (legend_text_position_x, legend_start_y + 60), 10, (0, 0, 255), -1)  # Blue dot
                cv2.putText(white_frame, 'Hands Landmark',
                            (legend_text_position_x + text_offset_x, legend_start_y + 65),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                # Add more legend items as needed
                cv2.imshow('MediaPipe Results', white_frame)

            else:
                text = f'Participant P03 | TASK: BIAH | Frame No: {frame_number:04d}'
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.45
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) - 350
                text_y = 30  # Set vertical position for text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 0, 250), thickness)

                cv2.imshow('Video Display', frame)

            # Display the resulting frame

            if cv2.waitKey(400) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# Usage example
video_path = '/home/iamshri/ml_projects/Datasets/QUB-PHEO/p03/CAM_LL/BIAH_BV.mp4'
visualizer = Visualizer(video_path, show_landmark=True)
# visualizer = Visualizer(video_path, show_landmark=False)
visualizer.visualize()
