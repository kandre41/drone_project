from ultralytics import YOLO
from utils.utils import center_crop
import cv2
import pandas as pd
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

df_list=[]
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
    xy = results[0].keypoints.xy.detach().cpu().numpy()[:,:13]  # x, y, visibility (if available)
    print(xy)
    print(xy.shape)
    df_list.append({
        "nose_x": xy[0,0,0],
        "nose_y": xy[0,0,1],
        "left_eye_x": xy[0,1,0],
        "left_eye_y": xy[0,1,1],
        "right_eye_x": xy[0,2,0],
        "right_eye_y": xy[0,2,1],
        "left_ear_x": xy[0,3,0],
        "left_ear_y": xy[0,3,1],
        "right_ear_x": xy[0,4,0],
        "right_ear_y": xy[0,4,1],
        "left_shoulder_x": xy[0,5,0],
        "left_shoulder_y": xy[0,5,1],
        "right_shoulder_x": xy[0,6,0],
        "right_shoulder_y": xy[0,6,1],
        "left_elbow_x": xy[0,7,0],
        "left_elbow_y": xy[0,7,1],
        "right_elbow_x": xy[0,8,0],
        "right_elbow_y": xy[0,8,1],
        "left_wrist_x": xy[0,9,0],
        "left_wrist_y": xy[0,9,1],
        "right_wrist_x": xy[0,10,0],
        "right_wrist_y": xy[0,10,1],
        "left_hip_x": xy[0,11,0],
        "left_hip_y": xy[0,11,1],
        "right_hip_x": xy[0,12,0],
        "right_hip_y": xy[0,12,1]
    })
    print(df_list)
    df=pd.DataFrame(df_list)
    print(df)

    cv2.waitKey(0)
    

# --- Cleanup ---
cv2.destroyAllWindows()  # Close all OpenCV display windows
del results
del model