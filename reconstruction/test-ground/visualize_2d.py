import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# Load HDF5 file
file_path = '/home/iamshri/Documents/Dataset/p01/CAM_LL/landmarks/BIAH_BV.hdf5'
hdf5_file = h5py.File(file_path, 'r')

# Extract datasets
face_landmarks = hdf5_file['face_landmarks'][:]
left_hand_landmarks = hdf5_file['left_hand_landmarks'][:]
pose_landmarks = hdf5_file['pose_landmarks'][:]
right_hand_landmarks = hdf5_file['right_hand_landmarks'][:]

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(8, 8))


def update(frame_idx):
    ax.clear()

    # Plot face landmarks
    face = face_landmarks[frame_idx]
    ax.scatter(face[:, 0], face[:, 1], c='red', s=20, label='Face Landmarks')

    # Plot left hand landmarks
    left_hand = left_hand_landmarks[frame_idx]
    ax.scatter(left_hand[:, 0], left_hand[:, 1], c='blue', s=20, label='Left Hand Landmarks')

    # Plot right hand landmarks
    right_hand = right_hand_landmarks[frame_idx]
    ax.scatter(right_hand[:, 0], right_hand[:, 1], c='green', s=20, label='Right Hand Landmarks')

    # Plot pose landmarks
    pose = pose_landmarks[frame_idx]
    ax.scatter(pose[:, 0], pose[:, 1], c='purple', s=20, label='Pose Landmarks')

    ax.legend()
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_aspect('equal')
    ax.set_title(f'Frame {frame_idx}')
    ax.axis('off')


# Create animation
ani = FuncAnimation(fig, update, frames=range(923), repeat=False)

# Save animation as video
ani.save('left_landmark_video-fps20-dpi300.mp4', writer='ffmpeg', fps=20, dpi=300)
