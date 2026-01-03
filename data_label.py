"""
-Start with a video
-frames will be extracted from the video
-machine learning mapping approach
-labels will be interpolated to [-1 to 1]
-interpolation points will be stored as a list of tuples for each of the four targets
-(start frame, end frame, start label, end label) and in between the frames, some interpolation will be used
-each data point coordinates (x,y) will be recentered based on the nose and scaled based on distance between shoulders
multi-target regression will have [throttle, pitch, roll, yaw]

"""

from ultralytics import YOLO
import os
import pandas as pd

path="W:\\VSCode\\drone_project"
frames_path='W:\\VSCode\\drone_project\\datasets\\images'

folder_name = 'demo1' #change the name of the folder containing the images within the \\images folder

model = YOLO(f'{path}\\weights\\yolo11x-pose.engine')

throttle = [()]
pitch = [()]
roll = [()]
yaw = [()]

results = model.predict(source=f'{frames_path}\\{folder_name}', half=True, device='cuda:0', stream=True)

output_dir = os.path.join(frames_path, f"{folder_name}_processed")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_list=[]

for result in results:
    original_name = os.path.basename(result.path) #needs basename because result.path is more than just the filename
    save_path = os.path.join(output_dir, f"processed_{original_name}")
    result.save(filename=save_path)

    xy = result.keypoints.xy.detach().cpu().numpy()[:,:13]
    #shape is person, 13 keypoints, (x y)
    """
    1- nose
    2- left_eye
    3- right_eye
    4- left_ear
    5- right_ear
    6- left_shoulder
    7- right_shoulder
    8- left_elbow
    9- right_elbow
    10- left_wrist
    11- right_wrist
    12- left_hip
    13- right_hip
    """
    if xy.shape==(1,13,2) or xy.shape==(2,13,2):
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

df=pd.DataFrame(df_list)
print(df.head())
print("df shape: ", df.shape)
    