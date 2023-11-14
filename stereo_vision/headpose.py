import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Thread
from queue import Queue


class HeadPose:
    def __init__(self, cap=0, calibration_data=None):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.calibration_data = calibration_data
        if self.calibration_data:
            self.camera_matrix, self.dist_coeffs, _, _ = self.load_calibration_data(calibration_data)
        else:
            self.camera_matrix, self.dist_coeffs = None, None
        self.cap = cv2.VideoCapture(cap)
        self.frame_queue = Queue(maxsize=2)
        self.pose_queue = Queue(maxsize=2)
        self.head_pose_indices = [33, 263, 1, 61, 291, 199]

    def load_calibration_data(self, calibration_data):
        with np.load(calibration_data) as data:
            mtx, dist, rvecs, tvecs = data['mtx'], data['dist'], data['rvecs'], data['tvecs']
            return mtx, dist, rvecs, tvecs

    def process_image(self, image):
        # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def process_image_thread(self):
        while True:
            success, image = self.cap.read()
            if not success:
                continue
            image, results = self.process_image(image)
            self.frame_queue.put((image, results))

    def calculate_head_pose_thread(self):
        while True:
            if not self.frame_queue.empty():
                image, results = self.frame_queue.get()
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        text, x, y, z, forehead_3d, forehead_2d, rot_vec, trans_vec, cam_matrix, dist_matrix = self.calculate_head_pose(
                            image, face_landmarks)
                        self.pose_queue.put((image, text, x, y, z, forehead_3d, forehead_2d, face_landmarks, rot_vec,
                                             trans_vec, cam_matrix, dist_matrix))

    def calculate_head_pose(self, image, face_landmarks):
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        forehead_3d = None
        forehead_2d = None

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in self.head_pose_indices:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
            if idx == 10:  # 151:
                forehead_2d = (lm.x * img_w, lm.y * img_h)
                forehead_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        if self.camera_matrix is not None and self.dist_coeffs is not None:
            cam_matrix = self.camera_matrix
            dist_matrix = self.dist_coeffs
        else:
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        print(f'x: {x} | y: {y} | z: {z}')

        if x < -1 and y < -1:
            text = "Looking Down-Left"
        elif x < -2 and y > 2:
            text = "Looking Down-Right"
        elif x > 1 and y < -1:
            text = "Looking Upper-Left"
        elif x > 1 and y > 1:
            text = "Looking Upper-Right"
        elif y < -1:
            text = "Looking Left"
        elif y > 1:
            text = "Looking Right"
        elif x < -1:
            text = "Looking Down"
        elif x > 1:
            text = "Looking Up"
        else:
            text = "Forward"

        return text, x, y, z, forehead_3d, forehead_2d, rot_vec, trans_vec, cam_matrix, dist_matrix

    def display_results(self, image, text, x, y, z, forehead_3d, forehead_2d, face_landmarks, rot_vec, trans_vec,
                        cam_matrix,
                        dist_matrix):
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # print(f'Forehead 3d: {forehead_3d}')
        forehead_3d_projection, _ = cv2.projectPoints(forehead_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p1 = (int(forehead_2d[0]), int(forehead_2d[1]))
        p2 = (int(forehead_2d[0] + y * 151), int(forehead_2d[1] - x * 151))
        cv2.line(image, p1, p2, (255, 0, 0), 3)

        mp.solutions.drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.drawing_spec)

    def run(self):
        # Initialize image to None
        image = None

        # Start the threads
        process_thread = Thread(target=self.process_image_thread)
        pose_thread = Thread(target=self.calculate_head_pose_thread)

        process_thread.start()
        pose_thread.start()

        while True:
            start_time = time.time()  # Record the start time

            if not self.pose_queue.empty():
                image, text, x, y, z, forehead_3d, forehead_2d, face_landmarks, rot_vec, trans_vec, cam_matrix, dist_matrix = self.pose_queue.get()
                self.display_results(image, text, x, y, z, forehead_3d, forehead_2d, face_landmarks, rot_vec, trans_vec,
                                     cam_matrix, dist_matrix)

                # Calculate FPS only when we have a valid image
                end = time.time()
                total_time = end - start_time
                fps = 1 / total_time
                cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                # display_size = (320, 180)
                display_size = (640, 480)
                resized_image = cv2.resize(image, display_size)
                # Show the image
                cv2.imshow('Head Pose', resized_image)

            if image is not None and cv2.waitKey(5) & 0xFF == 27:
                break

        # Close video capture
        self.cap.release()
        cv2.destroyAllWindows()

        # Stop the threads (optional)
        process_thread.join()
        pose_thread.join()

if __name__ == '__main__':
    video_direc = '/home/iamshri/Documents/Test-Video/p41/CAM_LL/GX010539_CC.avi'
    video_direc = '/home/iamshri/Documents/Test-Video/p41/CAM_LR/GX010508_CC.avi'
    calibration_file = '/home/iamshri/Documents/Test-Video/p41/CAM_LL/calib_param_CALIBRATION.npz'
    head_pose_estimator = HeadPose(cap=video_direc, calibration_data=calibration_file)
    head_pose_estimator.run()
    # if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    #     print('CUDA device found! Running on GPU.')
    # else:
    #     print('No CUDA device found, running on CPU.')


