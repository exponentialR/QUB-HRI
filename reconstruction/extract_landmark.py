import json
import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

from utils import load_calibration_data, get_video_fps, extract_landmarks, calculate_gaze_vector

class ExtractData:
    def __init__(self, left_video_path, right_video_path, left_path_h5=None, right_path_h5=None, fps=None, max_hands=2):

        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        self.left_path_hd5 = left_path_h5
        self.right_path_hd5 = right_path_h5

        # Lod participant video calibration data
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_tracking_confidence=.5, min_detection_confidence=.5,
                                                         refine_landmarks=True)
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=.5, min_tracking_confidence=.5,
                                              max_num_hands=max_hands)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=.5, min_tracking_confidence=.5)

        self.head_pose_indices = [33, 263, 1, 61, 291, 199]
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]

        if fps is None:
            self.left_fps = get_video_fps(self.left_video_path)
            self.right_fps = get_video_fps(self.right_video_path)
        else:
            self.fps = fps

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image

    def process_head_gaze_face(self, left_frame, right_frame, left_frame_index, right_frame_index):
        left_image, left_results = self.process_frame(left_frame)
        right_frame, right_results = self.process_frame(right_frame)

        left_timestamp = left_frame_index / self.fps
        right_timestamp = right_frame_index / self.fps
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            head_eye_data = self.calculate_head_pose(image, face_landmarks)
            gaze_vector = calculate_gaze_vector(head_eye_data['li3d'], head_eye_data['ri3d'], head_eye_data['f3d'],
                                                head_eye_data['rvec'])
            facial_landmarks = extract_landmarks(face_landmarks)

            # Convert NumPy arrays in head_eye_data to lists
            for key in ['f3d', 'f2d', 'rvec', 'tvec', 'li3d', 'ri3d', 'cam_matrix', 'dist_matrix']:
                if key in head_eye_data and isinstance(head_eye_data[key], np.ndarray):
                    head_eye_data[key] = head_eye_data[key].tolist()

            # Convert gaze_vector and facial_landmarks if they are NumPy arrays
            if isinstance(gaze_vector, np.ndarray):
                gaze_vector = gaze_vector.tolist()
            if isinstance(facial_landmarks, np.ndarray):
                facial_landmarks = facial_landmarks.tolist()
            face_present = 1
            data = {
                'timestamp': timestamp,
                'frame_index': frame_index,
                'headpose_data': head_eye_data,
                'gaze_2d': gaze_vector,
                'face_landmarks': facial_landmarks,
                'face_landmark_presence': face_present
            }

        else:  # in case there is no face landmark detected
            cam_matrix = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            hand_eye_data = {
                'head_pitch': 0, 'head_yaw': 0, 'head_roll': 0,
                'f3d': (0, 0, 0), 'f2d': (0, 0),
                'rvec': (0, 0, 0), 'tvec': (0, 0, 0),
                'cam_matrix': cam_matrix,
                'dist_matrix': dist_matrix,
                'li3d': (0, 0, 0), 'ri3d': (0, 0, 0),
                'text': 'Head-Eye'
            }
            gaze_vector = (0, 0, 0)
            facial_landmarks = [(0.0, 0.0, 0.0)] * 13
            face_present = 0
            data = {'timestamp': timestamp,
                    'frame_index': frame_index,
                    'headpose_data': hand_eye_data,
                    'gaze_2d': gaze_vector,
                    'face_landmarks': facial_landmarks,
                    'face_landmark_presence': face_present
                    }
        return data

    def calculate_head_pose(self, left_image, right_image, left_face_landmarks, right_face_landmarks):
        """
        Given an image and corresponding facial landmark,
        this method extracts head pose [pitch, yaw, and roll], left and right iris in 3D
        :param image: RGB image
        :param face_landmarks: 473 facial landmarks
        :return:
        """
        left_image_height, left_image_width, _ = left_image.shape
        right_image_height, right_image_width, _ = right_image.shape

        face_3d_left, face_2d_left = [], []
        face_3d_right, face_2d_right = [], []
        left_left_iris_3d, left_right_iris_3d = None, None
        right_left_iris_3d, right_right_iris_3d = None, None

        left_forehead_2d, left_forehead_3d = None, None
        right_forehead_2d, right_forehead_3d = None, None

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in self.head_pose_indices:
                x, y = int(lm.x * img_width), int(lm.y * img_height)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
            if idx == 6:
                forehead_2d = (lm.x * img_width, lm.y * img_height)
                forehead_3d = (lm.x * img_width, lm.y * img_height, lm.z)
            if idx == 468:
                left_iris_3d = (int(lm.x * img_width), int(lm.y * img_height), int(lm.z))
            if idx == 473:
                right_iris_3d = (int(lm.x * img_width), int(lm.y * img_height), int(lm.z * 3000))

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        if self.part_cam_matrix is not None and self.part_dist_coeffs is not None:
            cam_matrix = self.part_cam_matrix
            dist_matrix = self.part_dist_coeffs
        else:
            focal_length = 1 * img_width
            cam_matrix = np.array([[focal_length, 0, img_height / 2],
                                   [0, focal_length, img_width / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rvec)
        euler_angles = cv2.decomposeProjectionMatrix(rmat)[6]
        pitch, yaw, roll = euler_angles
        text = 'Head-Eye'
        head_eye = {
            # 'x': x, 'y': y, 'z': z,
            'head_pitch': pitch, 'head_yaw': yaw, 'head_roll': roll,
            'f3d': forehead_3d, 'f2d': forehead_2d,
            'rvec': rvec, 'tvec': tvec,
            'cam_matrix': cam_matrix,
            'dist_matrix': dist_matrix,
            'li3d': left_iris_3d, 'ri3d': right_iris_3d,
            'text': text
        }
        return head_eye

    def process_hand_landmarks(self, frame):
        results = self.hands.process(frame)
        hand_landmarks_list = []

        if results.multi_hand_landmarks:
            # Extract landmarks for the first detected hand
            for lm in results.multi_hand_landmarks[0].landmark:
                hand_landmarks_list.append((lm.x, lm.y, lm.z))
        else:
            # Append zeros for each of the 21 landmarks if no hand is detected
            hand_landmarks_list = [(0.0, 0.0, 0.0) for _ in range(21)]

        if isinstance(hand_landmarks_list, np.ndarray):
            hand_landmarks_list = hand_landmarks_list.tolist()

        return hand_landmarks_list

    def process_upper_body_pose(self, frame):
        upper_body_landmarks = []
        results = self.pose.process(frame)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in range(11, 24):
                    upper_body_landmarks.append((landmark.x, landmark.y, landmark.z))
        else:
            # If no landmarks detected, fill with zeros
            upper_body_landmarks = [(0.0, 0.0, 0.0)] * 13
        return upper_body_landmarks

    def extract_data(self):

        left_cap_participant = cv2.VideoCapture(self.left_video_path)
        left_video_directory = os.path.dirname(self.left_video_path)
        left_file_name = os.path.join(left_video_directory,
                                      os.path.splitext(os.path.basename(self.left_video_path))[0] + '.h5')
        # TODO:

        total_frames = int(left_cap_participant.get(cv2.CAP_PROP_FRAME_COUNT))

        with h5py.File(left_file_name, 'w') as hdf_file:
            pbar = tqdm(total=total_frames, desc='Extracting Landmarks from Video', unit='frame')
            timestamps, frame_indices, gaze_2d, face_landmarks, hand_landmarks, upper_body_pose, face_presence = [], [], [], [], [], [], []
            headpose_data = {key: [] for key in
                             ['head_pitch', 'head_yaw', 'head_roll', 'f3d', 'f2d', 'rvec', 'tvec', 'cam_matrix',
                              'dist_matrix', 'li3d', 'ri3d']}

            while left_cap_participant.isOpened():
                ret, frame = left_cap_participant.read()
                if not ret:
                    break
                try:
                    data = self.process_head_gaze_face(frame, pbar.n)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    hand_landmark = self.process_hand_landmarks(image)
                    upper_body_landmarks = self.process_upper_body_pose(image)

                    image.flags.writeable = True
                    # if data:
                    timestamps.append(data['timestamp'])
                    frame_indices.append(data['frame_index'])
                    for key in headpose_data:
                        headpose_data[key].append(data['headpose_data'][key])
                    gaze_2d.append(data['gaze_2d'])
                    face_landmarks.append(data['face_landmarks'])

                    # hand_landmarks.append(data['hand_landmarks'])
                    upper_body_pose.append(upper_body_landmarks)
                    hand_landmarks.append(hand_landmark)
                    face_presence.append(data['face_landmark_presence'])

                except Exception as e:
                    print(f"Error processing frame {pbar.n}: {e}")
                pbar.update(1)

            left_cap_participant.release()

            # Save data to HDF5
            hdf_file.create_dataset('timestamps', data=np.array(timestamps))
            hdf_file.create_dataset('face_landmark_presence', data=np.array(face_presence))
            hdf_file.create_dataset('frame_indices', data=np.array(frame_indices))
            for key, values in headpose_data.items():
                hdf_file.create_dataset(f'headpose_data/{key}', data=np.array(values))
            hdf_file.create_dataset('gaze_2d', data=np.array(gaze_2d))
            hdf_file.create_dataset('face_landmarks', data=np.array(face_landmarks))
            hdf_file.create_dataset('upper_body_pose', data=np.array(upper_body_pose))
            hdf_file.create_dataset('hand_landmarks', data=np.array(hand_landmarks))

    def read_h5_file(self):
        with h5py.File(self.left_path_hd5, 'r') as hdf_file:
            timestamps = hdf_file['timestamps'][:]
            timestamps_list = timestamps.tolist()
            print(f'Length of timestamps: {len(timestamps_list)}')
            data = {}
            for key in hdf_file.keys():
                if isinstance(hdf_file[key], h5py.Group):
                    group_data = {}
                    for subkey in hdf_file[key].keys():
                        group_data[subkey] = hdf_file[key][subkey][:]
                    data[key] = group_data
                elif isinstance(hdf_file[key], h5py.Dataset):
                    data[key] = hdf_file[key][:]

        return data
