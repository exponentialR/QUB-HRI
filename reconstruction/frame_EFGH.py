import numpy as np
import cv2
import mediapipe as mp
import os

from utils import get_video_fps, extract_landmarks, calculate_gaze_vector
os.environ["GLOG_minloglevel"] ="2"



class FrameEFGH:
    def __init__(self, image, image_index, cam_matrix, dist_coeffs, fps=None, max_hands=2):
        self.image = image
        self.image_index = image_index
        self.cam_matrix, self.dist_coeffs = cam_matrix, dist_coeffs
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_tracking_confidence=.5, min_detection_confidence=.5,
                                                         refine_landmarks=True)
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5,
                                              max_num_hands=max_hands)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=.5, min_tracking_confidence=.5)

        self.head_pose_indices = [33, 263, 1, 61, 291, 199]
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]
        self.fps = fps

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image

    def calculate_head_pose(self, processed_image, face_landmarks):
        """
        Given an image and corresponding facial landmark,
        this method extracts head pose [pitch, yaw, and roll], Iris in 3D and 2D, and forehead in 3D and 2D
        :param processed_image: Mediapipe processed image
        :param face_landmarks: 473 facial landmarks
        :return:
        """
        processed_image_height, processed_image_width, _ = processed_image.shape
        face, face_2d = [], []
        left_iris, right_iris = None, None
        forehead = None

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in self.head_pose_indices:
                x, y = int(lm.x * processed_image_width), int(lm.y * processed_image_height)
                face.append([x, y, lm.z])
                face_2d.append([x, y])
            if idx == 6:
                forehead = (lm.x * processed_image_width, lm.y * processed_image_height, int(lm.z))
            if idx == 468:
                left_iris = (int(lm.x * processed_image_width), int(lm.y * processed_image_height), int(lm.z))
            if idx == 473:
                right_iris = (int(lm.x * processed_image_width), int(lm.y * processed_image_height), int(lm.z))

        face_extracted = np.array(face, dtype=np.float64)
        face_2d = np.array(face_2d, dtype=np.float64)

        if self.cam_matrix is not None and self.dist_coeffs is not None:
            cam_matrix = self.cam_matrix
            dist_matrix = self.dist_coeffs
        else:
            focal_length = 1 * processed_image_width
            cam_matrix = np.array([[focal_length, 0, processed_image_height / 2],
                                   [0, focal_length, processed_image_width / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
        # print(f"Number of 3D points: {len(face_extracted)}")
        # print(f"Number of 2D points: {len(face_2d)}")

        success, rvec, tvec = cv2.solvePnP(face_extracted, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        # euler_angles = cv2.decomposeProjectionMatrix(rmat)[6]
        pitch, yaw, roll = euler_angles.flatten()[:3]
        text = 'HeadEyeData'
        head_eye = {
            'head_pitch': pitch, 'head_yaw': yaw, 'head_roll': roll, 'forehead': forehead,
            'rvec': rvec, 'tvec': tvec,
            'cam_matrix': cam_matrix,
            'dist_matrix': dist_matrix,
            'left_iris': left_iris, 'right_iris': right_iris,
            'text': text
        }
        return head_eye

    def processEFGH(self):
        """
        Given a frame, this method processes the frame and extracts the following data:
        - Timestamp of the frame
        - Face presence
        - Head pose [pitch, yaw, roll]
        - Gaze vector
        - Facial landmarks
        - Hand landmarks

        :param frame:
        :param frame_index:
        :return:
        """
        face_results, processed_image = self.process_frame(self.image)
        timestamp = self.image_index/self.fps
        # Process Hand Pose
        hand_results = self.hands.process(processed_image)
        hand_landmarks_list = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_landmarks = extract_landmarks(hand_landmarks)
                hand_landmarks_list.append(hand_landmarks)
        else:
            hand_landmarks_list = None
        if isinstance(hand_landmarks_list, np.ndarray):
            hand_landmarks_list = hand_landmarks_list.tolist()

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            head_eye_data = self.calculate_head_pose(processed_image, face_landmarks)
            gaze_vector = calculate_gaze_vector(head_eye_data['left_iris'], head_eye_data['right_iris'],
                                                 head_eye_data['forehead'], head_eye_data['rvec'])
            facial_landmarks = extract_landmarks(face_landmarks)

            for key in head_eye_data.keys():
                if key in head_eye_data and isinstance(head_eye_data[key], np.ndarray):
                    head_eye_data[key] = head_eye_data[key].tolist()

            if isinstance(gaze_vector, np.ndarray):
                gaze_vector = gaze_vector.tolist()
            if isinstance(facial_landmarks, np.ndarray):
                facial_landmarks = facial_landmarks.tolist()
            face_present = 1

        else:
            face_present = 0
            head_eye_data = None
            gaze_vector = None
            facial_landmarks = None

        data = {
            'timestamp': timestamp,
            'frame_index': self.image_index,
            'face_present': face_present,
            'head_eye_data': head_eye_data,
            'gaze_vector': gaze_vector,
            'facial_landmarks': facial_landmarks,
            'hand_landmarks': hand_landmarks_list
        }

        return data






