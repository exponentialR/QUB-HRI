import os.path
import os
import numpy as np
import cv2


def detect_charuco_board(frame, aruco_dict, board, min_corners):
    """
    Detects a Charuco board in the given frame
    Returns True if the board is detected with at least min_corners detected: otherwise Returns False
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_corners is not None and len(charuco_corners) >= min_corners:
            return True
    return False


def create_dir(directory):
    """Utility function to create directory if it doesn't exist.

    Parameters:
        - directory: The directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def remove_files_in_folder(folder_path, file_ext=('png', 'jpg', 'jpeg')):
    """
   Remove all files in a specified folder path

   :param folder_path: Path to the folder from which files will be deleted
   :param file_ext: Tuple of file extensions to delete
   :return:
   """
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path) and file.endswith((file_ext)):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"Skipped deleting: {file_path} as it's not a file.")
        except Exception as e:
            print(f"Failed to delete: {file_path} due to {e}")


def extract_landmarks(face_landmarks):
    flattened_landmarks = []
    for lm in face_landmarks.landmark:
        flattened_landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(flattened_landmarks)


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def convert_to_structured_array(face_landmarks):
    # Define the structured array data format
    dtypes = [(key, 'f8', (3,)) for key in face_landmarks.keys()]
    structured_array = np.zeros((1,), dtype=dtypes)

    # Fill the structured array with data
    for key, value in face_landmarks.items():
        structured_array[key] = value

    return structured_array
