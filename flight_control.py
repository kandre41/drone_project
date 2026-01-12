from ultralytics import YOLO
from utils.utils import keypoint_mapper, feature_engineer, MovingAverage
import torch
from models import PoseControlNet
import cv2
import pandas as pd
import os
import joblib
import numpy as np
from DJITelloPy.djitellopy import tello
import time
PATH=r"W:\VSCode\drone_project"

# 0 refers to the default primary camera 
WEBCAM_SOURCE = 0

drone=tello.Tello()
drone.connect()
time.sleep(10)
def main():
    loaded_scaler = joblib.load(os.path.join(PATH,r"scaler\scaler.bin"))
    model = YOLO(os.path.join(PATH,r"weights\yolo11x-pose.engine"))
    mlp = PoseControlNet(num_controls=4)
    mlp.load_state_dict(torch.load(os.path.join(PATH,r"weights\pose_control_model.pt"), weights_only=True))
    mlp.eval()
    cap = cv2.VideoCapture(WEBCAM_SOURCE) #sets cap to the webcam source
    #cap=cv2.VideoCapture(os.path.join(PATH,r"datasets\videos\demo1.mp4"))
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Webcam successfully opened. Press 'q' to exit.")
    take_off=False
    throttle_queue = MovingAverage(5)
    pitch_queue = MovingAverage(5)
    roll_queue = MovingAverage(5)
    yaw_queue = MovingAverage(5)
    f_throttle = lambda x: x if abs(x) > 0.05 else 0
    f_pitch = lambda x: x if abs(x) > 0.05 else 0
    f_roll = lambda x: x if abs(x) > 0.05 else 0
    f_yaw = lambda x: x if abs(x) > 0.2 else 0
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
                df = feature_engineer(df)
                print(df.loc[0,'left_elbow_angle'])
                if df.loc[0,'left_elbow_angle']<-2.5 and not take_off:
                    take_off=True
                    drone.takeoff()
                if take_off:
                    dist_cols=df.filter(like='len').columns
                    df[x_columns] = df[x_columns].sub((df['right_shoulder_x']+df['left_shoulder_x'])/2,axis=0)
                    df[y_columns] = df[y_columns].sub((df['right_shoulder_y']+df['left_shoulder_y'])/2,axis=0)
                    shoulder_dist=np.sqrt((df['right_shoulder_x']-df['left_shoulder_x'])**2+
                                            (df['right_shoulder_y']-df['left_shoulder_y'])**2)
                    # Replace 0 or NaN with the average to prevent math errors
                    avg_dist = shoulder_dist.median()
                    shoulder_dist = shoulder_dist.replace(0, np.nan).fillna(avg_dist)
                    df[x_columns] = df[x_columns].div(shoulder_dist, axis=0)
                    df[y_columns] = df[y_columns].div(shoulder_dist, axis=0)
                    df[dist_cols] = df[dist_cols].div(shoulder_dist, axis=0)

                    result_features=df.values
                    scaled_result_features=loaded_scaler.transform(result_features.reshape(1,46))
                    result_tensor=torch.tensor(scaled_result_features, dtype=torch.float32).reshape((1,46))
                    control_vector=mlp(result_tensor).detach().flatten().numpy() #using cpu inference seems faster than transfering to and back from the gpu
                    
                    throttle = control_vector[0]
                    pitch = control_vector[1]
                    roll = control_vector[2]
                    yaw = control_vector[3]
                    
                    print(throttle, pitch, roll, yaw)

                    throttle = f_throttle(throttle)
                    pitch = f_pitch(pitch)
                    roll = f_roll(roll)
                    if abs(pitch) > 0.15: #camera angle makes it hard for MLP to decouple the pitch from the roll at higher values
                        roll = 0 
                    yaw = f_yaw(yaw)

                    avg_throttle = throttle_queue.add_value(throttle)
                    avg_pitch = pitch_queue.add_value(pitch)
                    avg_roll = roll_queue.add_value(roll)
                    avg_yaw = yaw_queue.add_value(yaw)

                    print(avg_throttle, avg_pitch, avg_roll, avg_yaw)
                    drone.send_rc_control(int(avg_roll*0), int(avg_pitch*60), int(avg_throttle*100), int(avg_yaw*0))
                    print(df.loc[0,'left_arm_angle'])
                    if df.loc[0,'left_arm_angle']<-2.2 and take_off:
                        drone.land()
                        take_off=False
                        break                        
            else:
                drone.land()
                time.sleep(2)
                break
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