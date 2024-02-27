import cv2
import numpy as np

FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL

class DepthEstimation:
    def __init__(self, extrinsic_path, left_video, right_video):
        self.left_extrinsic_path = extrinsic_path
        self.left_video = left_video
        self.right_video = right_video
        self.left_cap = cv2.VideoCapture(left_video)
        self.right_cap = cv2.VideoCapture(right_video)

        self.extrinsic_parameters = np.load(extrinsic_path)
        self.left_camera_matrix, self.left_dist_coeffs = self.extrinsic_parameters['left_mtx'], self.extrinsic_parameters[
            'left_dist']
        self.right_camera_matrix, self.right_dist_coeffs = self.extrinsic_parameters['right_mtx'], \
        self.extrinsic_parameters['right_dist']
        self.R = self.extrinsic_parameters['R']
        self.T = self.extrinsic_parameters['T']

        self.image_size = (
        int(self.left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.focal_length = self.left_camera_matrix[0, 0]
        self.baseline = np.linalg.norm(self.T)
        print(self.baseline)

    def calculate_depth(self, left_frame, right_frame):
        pass

    def rectify_images(self, left_frame, right_frame):
        # Pre-filtering frames for noise reduction anqd edge preservation

        # Your existing rectification process
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(self.left_camera_matrix, self.left_dist_coeffs,
                                                    self.right_camera_matrix, self.right_dist_coeffs,
                                                    self.image_size, self.R, self.T, alpha=0.5)
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(self.left_camera_matrix, self.left_dist_coeffs, R1, P1,
                                                             self.image_size, cv2.CV_32FC1)
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(self.right_camera_matrix, self.right_dist_coeffs, R2, P2,
                                                               self.image_size, cv2.CV_32FC1)
        left_rectified = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LANCZOS4)
        right_rectified = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LANCZOS4)
        return left_rectified, right_rectified

    def depth_compute(self, left_frame, right_frame):
        # left_rectified, right_rectified = self.rectify_images(left_frame, right_frame)

        stereo = cv2.StereoSGBM_create(minDisparity=6, numDisparities=16, blockSize=10)

        # stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=5, P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
        #                                disp12MaxDiff=1, preFilterCap=63, uniquenessRatio=10, speckleWindowSize=100,
        #                                speckleRange=16)

        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=25)
        # disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
        disparity = stereo.compute(left_frame, right_frame, cv2.CV_32F)

        with np.errstate(divide='ignore'):
            depth_map = (self.focal_length * self.baseline) / (disparity + (disparity == 0))
        return depth_map

    def run(self):
        count = 0
        while True:
            left_ret, left_frame = self.left_cap.read()
            right_ret, right_frame = self.right_cap.read()
            if not left_ret or not right_ret:
                break

            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            depth_map = self.depth_compute(left_frame, right_frame)
            depth_map_normalized = cv2.normalize(depth_map, None, alpha=1.5, beta=255, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_8U)

            depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)


            display_size = (640, 480)
            depth_map_resized = cv2.resize(depth_map_colored, display_size)
            right_frame_resized = cv2.resize(right_frame, display_size)

            # Convert left frame to BGR format if it's grayscale
            if len(right_frame_resized.shape) == 2:
                right_frame_resized = cv2.cvtColor(right_frame_resized, cv2.COLOR_GRAY2BGR)

            # Stack resized frames horizontally
            stacked_frames = np.hstack((right_frame_resized, depth_map_resized))

            if count % 10 == 0:
                cv2.imshow('Depth Map', stacked_frames)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.left_cap.release()
        self.right_cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    extrinsic_path = '/home/qub-hri/Documents/QUBVisionData/RawData/stereo/stereo_calib.npz'
    left_video = '/home/qub-hri/Documents/QUBVisionData/RawData/stereo/CAM_UL/CALIBRATION.mp4'
    right_video = '/home/qub-hri/Documents/QUBVisionData/RawData/stereo/CAM_UR/CALIBRATION.mp4'
    depth_estimation = DepthEstimation(extrinsic_path, left_video, right_video)
    depth_estimation.run()
    pass
