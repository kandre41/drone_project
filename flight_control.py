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
    mlp = PoseControlNet(num_controls=4)
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
                df=pd.DataFrame(keypoint_mapper(xy),index=[0])
                x_columns=df.filter(like='_x').columns
                y_columns=df.filter(like='_y').columns
                df[x_columns] = df[x_columns].sub((df['right_shoulder_x']+df['left_shoulder_x'])/2,axis=0)
                df[y_columns] = df[y_columns].sub((df['right_shoulder_y']+df['left_shoulder_y'])/2,axis=0)
                shoulder_dist=np.sqrt((df['right_shoulder_x']-df['left_shoulder_x'])**2+
                                        (df['right_shoulder_y']-df['left_shoulder_y'])**2)
                # Replace 0 or NaN with the average to prevent math errors
                avg_dist = shoulder_dist.median()
                shoulder_dist = shoulder_dist.replace(0, np.nan).fillna(avg_dist)
                df[x_columns] = df[x_columns].div(shoulder_dist,axis=0)
                df[y_columns] = df[y_columns].div(shoulder_dist,axis=0)

                result_features=df.values
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