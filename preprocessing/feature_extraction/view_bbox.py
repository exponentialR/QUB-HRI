from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data='config.yaml', epochs=100, imgsz=640)
trained_model = '/home/samueladebayo/PycharmProjects/LegoDetect/runs/detect/train8/weights/Lego_YOLO.pt'



import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(trained_model)

# Open the video file
video_path = "../../../Documents/PhD/QUBPHEO/Annotated/RPS/p03-CAM_AV-TOWER_HO-RPS-16.711147873569484_19.998175458310627.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
frame_number = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            x, y, w, h = box.xywhn[0].tolist()
            class_idx = int(box.cls)
            print(f" Frame Number: {frame_number} | Bounding Box: {x, y, w, h} | class ID: {class_idx} | Class Name: {model.names[class_idx]}")

        frame_number += 1

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(200) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()