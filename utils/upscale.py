import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(None, None, 3), pooling=None)

def upscale_frame(frame):
    frame = tf.cast(frame, tf.float32)/255.0
    frame = tf.image.resize(frame, (3840, 2160))

    upscaled_frame = model.predict(np.expand_dims(frame, axis=0))

    upscaled_frame = np.squeeze(upscaled_frame, axis=0)
    upscaled_frame = (upscaled_frame * 255).astype(np.uint8)
    return upscaled_frame

video_dir = '/home/iamshri/PycharmProjects/QUB-HRI/data/CALIBRATION.MP4'
cap = cv2.VideoCapture(video_dir)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('upscaled_video.mp4', fourcc, 30.0, (3840, 2160))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Upscale the frame
    upscaled_frame = upscale_frame(frame)

    # Write the upscaled frame
    out.write(upscaled_frame)

# Release everything when done
cap.release()
out.release()
