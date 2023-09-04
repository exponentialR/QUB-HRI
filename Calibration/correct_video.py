import cv2
import numpy as np
import glob


def load_calibration_parameters(load_path):
    data = np.load(load_path)
    return data['mtx'], data['dist']


def correct_video(video_path, mtx, dist, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected = cv2.undistort(frame, mtx, dist, None, mtx)
        out.write(corrected)

    cap.release()
    out.release()


if __name__ == "__main__":
    mtx, dist = load_calibration_parameters('calibration_parameters.npz')
    video_paths = glob.glob('videos_to_correct/*.mp4')

    for video_path in video_paths:
        save_path = f'corrected_videos/{video_path.split("/")[-1]}'
        correct_video(video_path, mtx, dist, save_path)
