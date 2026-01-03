from ultralytics import YOLO
from utils.utils import center_crop
import cv2

# 1. Load the YOLO model 
path="W:\\VSCode\\drone_project"
image_path="datasets\\images"
image="setup_test.jpg"
#model = YOLO(f'{path}\weights\\yolo11x-pose.engine') 
model = YOLO(f'{path}\\weights\\yolo11x-pose.engine')

# 2. Set the Webcam Source
# '0' typically refers to the default primary camera 
WEBCAM_SOURCE = 0

# --- Start Webcam Stream ---
# VideoCapture object to read frames from the camera
stream=False
if stream:
    cap = cv2.VideoCapture(WEBCAM_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Webcam successfully opened. Press 'q' to exit.")
else: 
    cap = cv2.imread(f'{path}\{image_path}\{image}')
    cap = center_crop(cap,(640,640))
# Check if the webcam was opened or if image was loaded


# --- Live Inference Loop ---
if stream:
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()  # Release the webcam capture object
else:
    
    results=model(cap)
    annotated_frame=results[0].plot()
    cv2.imshow('Image', annotated_frame)
    xyn = results[0].keypoints.xyn.detach().cpu().numpy()[:,:11]  # x, y, visibility (if available)
    print(xyn)
    cv2.waitKey(0)
    

# --- Cleanup ---
cv2.destroyAllWindows()  # Close all OpenCV display windows
del results
del model