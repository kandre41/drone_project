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
from utils import interpolater, keypoint_mapper, feature_engineer
from ultralytics import YOLO
import os
import pandas as pd
import numpy as np

path=r"W:\VSCode\drone_project"
frames_path=r'W:\VSCode\drone_project\datasets\images'

folder_name = 'demo8' #change the name of the folder containing the images within the \images folder

model = YOLO(f'{path}\\weights\\yolo11x-pose.engine')
start_frame=1
end_frame=636
throttle = [(311,0,-1,327),(333,-1,0,346),(355,0,1,372),(376,1,0,390),(400,0,-1,416),(421,-1,0,434),(440,0,1,458),(460,1,0,478)]
pitch = [(53,0,-1,80),(92,-1,0,114),(122,0,-1,147),(159,-1,0,180),(188,0,-1,209),(221,-1,0,238),(257,0,-1,279),(282,-1,0,297),(590,0,-1,604),(610,-1,0,625)]#p
roll = [(1,0,0,636)] #r
yaw = [(494,0,1,517),(522,1,0,538),(545,0,-1,559),(567,-1,0,578)]

results = model.predict(source=f'{frames_path}\\{folder_name}', half=True, device='cuda:0', stream=True)

output_dir = os.path.join(frames_path, f"{folder_name}_processed")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_list=[]

for result in results:
    original_name = os.path.basename(result.path) #needs basename because result.path is more than just the filename
    save_path = os.path.join(output_dir, f"processed_{original_name}")
    result.save(filename=save_path)

    xy = result.keypoints.data.detach().cpu().numpy()[:,:13]
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
    if xy.shape==(1,13,3) or xy.shape==(2,13,3): #a few frames had two people detected by the model
        df_list.append(keypoint_mapper(xy))

df=pd.DataFrame(df_list)
print(df.head())
print("df shape: ", df.shape)

frame_count = df.shape[0]
#normalization and scaling process

x_columns=df.filter(like='_x').columns
y_columns=df.filter(like='_y').columns
df = feature_engineer(df)
dist_cols=df.filter(like='len').columns

df[x_columns] = df[x_columns].sub((df['right_shoulder_x']+df['left_shoulder_x'])/2,axis=0)
df[y_columns] = df[y_columns].sub((df['right_shoulder_y']+df['left_shoulder_y'])/2,axis=0)
shoulder_dist=np.sqrt((df['right_shoulder_x']-df['left_shoulder_x'])**2+
                          (df['right_shoulder_y']-df['left_shoulder_y'])**2)
# Replace 0 or NaN with the average to prevent math errors
avg_dist = shoulder_dist.median()
shoulder_dist = shoulder_dist.replace(0, np.nan).fillna(avg_dist)
df[x_columns] = df[x_columns].div(shoulder_dist,axis=0)
df[y_columns] = df[y_columns].div(shoulder_dist,axis=0)
df[dist_cols] = df[dist_cols].div(shoulder_dist,axis=0)
#adding labels which need interpolation and also ffill and bfill 
throttle_arr = np.full(frame_count,np.nan)
pitch_arr = np.full(frame_count,np.nan)
roll_arr = np.full(frame_count,np.nan)
yaw_arr = np.full(frame_count,np.nan)
#convert to series and fill nan
throttle_series = pd.Series(interpolater(throttle, throttle_arr)).ffill().bfill()
pitch_series = pd.Series(interpolater(pitch,pitch_arr)).ffill().bfill()
roll_series = pd.Series(interpolater(roll, roll_arr)).ffill().bfill()
yaw_series = pd.Series(interpolater(yaw, yaw_arr)).ffill().bfill()

df['target_throttle'] = throttle_series.astype(np.float32)
df['target_pitch'] = pitch_series.astype(np.float32)
df['target_roll'] = roll_series.astype(np.float32)
df['target_yaw'] = yaw_series.astype(np.float32)
df=df.loc[start_frame:end_frame-1]
print(df.head())

df.to_parquet(path=os.path.join(path,'datasets','labeled_data',f"{folder_name}.parquet"))
df.to_csv(os.path.join(path,'datasets','labeled_data',f"{folder_name}.csv"), index=False) #csv just to visually inspect