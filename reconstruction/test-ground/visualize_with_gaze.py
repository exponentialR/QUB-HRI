import cv2
import mediapipe as mp
import numpy as np

object_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right Mouth corner
], dtype="double")  # These values are example metrics and should be adjusted to better fit the facial metric data used

axis = np.float32([
    [500, 0, 0],  # x-axis point (red)
    [0, 500, 0],  # y-axis point (green)
    [0, 0, 500]  # z-axis point (blue)
]).reshape(-1, 3)


import cv2
import mediapipe as mp
import numpy as np

class Visualizer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils

        # Camera matrix, assume a center and focal length
        focal_length = 960
        center = (480, 270)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )

        # Assuming a negligible lens distortion
        self.dist_coeffs = np.zeros((4, 1))

    def visualize(self):
        cap = cv2.VideoCapture(self.video_path)
        landmark_indices = [1, 152, 234, 454, 356, 276]  # Example indices for nose, chin, eyes, and mouth corners

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (960, 540))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    image_points = np.array([
                        (face_landmarks.landmark[mp.solutions.face_mesh.FACEMESH_NOSE].x * frame.shape[1],
                         face_landmarks.landmark[mp.solutions.face_mesh.FACEMESH_NOSE].y * frame.shape[0]), # Nose tip
                        # Add other points similarly
                    ], dtype="double")

                    # Solve for pose
                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        object_points, image_points, self.camera_matrix, self.dist_coeffs
                    )

                    # Draw axis
                    axis_points, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)
                    frame = self.draw_axis(frame, image_points[0], axis_points)

            cv2.imshow('MediaPipe Results', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_axis(self, img, nose_tip, axis_points):
        # Draw the axes lines
        p_nose = (int(nose_tip[0]), int(nose_tip[1]))
        p_x = (int(axis_points[0][0][0]), int(axis_points[0][0][1]))
        p_y = (int(axis_points[1][0][0]), int(axis_points[1][0][1]))
        p_z = (int(axis_points[2][0][0]), int(axis_points[2][0][1]))

        img = cv2.line(img, p_nose, p_x, (0,0,255), 3)
        img = cv2.line(img, p_nose, p_y, (0,255,0), 3)
        img = cv2.line(img, p_nose, p_z, (255,0,0), 3)
        return img

# Usage
video_path = '/path/to/your/video.mp4'
visualizer = Visualizer(video_path)
visualizer.visualize()


# Usage
video_path = '/home/iamshri/PycharmProjects/QUB-HRI/preprocessing/data/output/output.mp4'
visualizer = Visualizer(video_path)
visualizer.visualize()
