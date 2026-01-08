from ultralytics import YOLO
from utils.utils import center_crop, keypoint_mapper
import torch
from models import PoseControlNet
import cv2
import pandas as pd
import os
import joblib
import numpy as np
# 1. Load the YOLO model 
PATH=r"W:\VSCode\drone_project"


#Set the Webcam Source
# '0' typically refers to the default primary camera 
WEBCAM_SOURCE = 1


def main():
    loaded_scaler = joblib.load(os.path.join(PATH,r"scaler\scaler.bin"))
    model = YOLO(os.path.join(PATH,r"weights\yolo11x-pose.engine"))
    mlp = PoseControlNet(num_controls=1)
    mlp.load_state_dict(torch.load(os.path.join(PATH,r"weights\pose_control_model.pt"), weights_only=True))
    mlp.eval()
    #cap = cv2.VideoCapture(WEBCAM_SOURCE) #sets cap to the webcam source
    cap=cv2.VideoCapture(os.path.join(PATH,r"datasets\videos\demo1.mp4"))
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Webcam successfully opened. Press 'q' to exit.")
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            results = model(frame)
            xy = results[0].keypoints.data.detach().cpu().numpy()[:,:13] # x, y, visibility (if available)
            if xy.shape == (1,13,3):
                result_features=np.array(list(keypoint_mapper(xy).values()))
                scaled_result_features=loaded_scaler.transform(result_features.reshape(1,39))
                result_tensor=torch.tensor(scaled_result_features, dtype=torch.float32).reshape((1,39))
                control_vector=mlp(result_tensor)

                print(control_vector)

            annotated_frame = results[0].plot()
            cv2.imshow("drone control", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()  # Release the webcam capture object
    cv2.destroyAllWindows()  # Close all OpenCV display windows
    del results
    del model
    
if __name__ == "__main__":
    main()