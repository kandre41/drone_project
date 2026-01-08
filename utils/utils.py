import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
def center_crop(img,size): #center crops and downsamples to lower, square resolution that yolo expects

    h,w=img.shape[:2] #y,x

    min_dim=min(h,w)
    max_dim=max(h,w)
    square_dim_half=(max_dim-min_dim)//2

    if min_dim==h: #landscape
        x_min=square_dim_half
        x_max=w-square_dim_half
        return cv2.resize(img[:,x_min:x_max],size,interpolation=cv2.INTER_AREA)
    else:
        y_min=square_dim_half
        y_max=h-square_dim_half
        return cv2.resize(img[y_min:y_max,:],size,interpolation=cv2.INTER_AREA)
def euclidean_distance(x1,x2,y1,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def interpolater(segments: list, arr: np.array) -> np.array:
    for start_i, start_v, end_v, end_i in segments:
        values_to_add = np.linspace(start_v, end_v, end_i-start_i+1)
        arr[start_i-1:end_i] = values_to_add
    return arr

def keypoint_mapper(xy):
    return {
        "nose_x": xy[0,0,0],
        "nose_y": xy[0,0,1],
        "nose_conf": xy[0,0,2],
        "left_eye_x": xy[0,1,0],
        "left_eye_y": xy[0,1,1],
        "left_eye_conf": xy[0,1,2],
        "right_eye_x": xy[0,2,0],
        "right_eye_y": xy[0,2,1],
        "right_eye_conf": xy[0,2,2],
        "left_ear_x": xy[0,3,0],
        "left_ear_y": xy[0,3,1],
        "left_ear_conf": xy[0,3,2],
        "right_ear_x": xy[0,4,0],
        "right_ear_y": xy[0,4,1],
        "right_ear_conf": xy[0,4,2],
        "left_shoulder_x": xy[0,5,0],
        "left_shoulder_y": xy[0,5,1],
        "left_shoulder_conf": xy[0,5,2],
        "right_shoulder_x": xy[0,6,0],
        "right_shoulder_y": xy[0,6,1],
        "right_shoulder_conf": xy[0,6,2],
        "left_elbow_x": xy[0,7,0],
        "left_elbow_y": xy[0,7,1],
        "left_elbow_conf": xy[0,7,2],
        "right_elbow_x": xy[0,8,0],
        "right_elbow_y": xy[0,8,1],
        "right_elbow_conf": xy[0,8,2],
        "left_wrist_x": xy[0,9,0],
        "left_wrist_y": xy[0,9,1],
        "left_wrist_conf": xy[0,9,2],
        "right_wrist_x": xy[0,10,0],
        "right_wrist_y": xy[0,10,1],
        "right_wrist_conf": xy[0,10,2],
        "left_hip_x": xy[0,11,0],
        "left_hip_y": xy[0,11,1],
        "left_hip_conf": xy[0,11,2],
        "right_hip_x": xy[0,12,0],
        "right_hip_y": xy[0,12,1],
        "right_hip_conf": xy[0,12,2]
    }
def plotter(train_loss, val_loss):
    epochs=list(range(len(train_loss)))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epochs',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    fig.show()
def signed_angle(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    angle_ba = np.arctan2(p1_y - p2_y, p1_x - p2_x)
    angle_bc = np.arctan2(p3_y - p2_y, p3_x - p2_x)
    
    delta_angle = angle_bc - angle_ba
        
    return delta_angle
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df['left_forearm_len'] = euclidean_distance(df['left_wrist_x'],df['left_elbow_x'],df['left_wrist_y'],df['left_elbow_y'])
    df['right_forearm_len'] = euclidean_distance(df['right_wrist_x'],df['right_elbow_x'],df['right_wrist_y'],df['right_elbow_y'])
    df['right_upper_arm_len'] = euclidean_distance(df['right_shoulder_x'],df['right_elbow_x'],df['right_shoulder_y'],df['right_elbow_y'])
    df['right_elbow_angle'] = signed_angle(df['right_wrist_x'], df['right_wrist_y'], 
                                         df['right_elbow_x'], df['right_elbow_y'],
                                         df['right_shoulder_x'], df['right_shoulder_y'])
    df['left_elbow_angle'] = signed_angle(df['left_wrist_x'], df['left_wrist_y'], 
                                         df['left_elbow_x'], df['left_elbow_y'],
                                         df['left_shoulder_x'], df['left_shoulder_y'])
    df['right_arm_angle'] = signed_angle(df['right_shoulder_x'],df['right_shoulder_y'],
                                          df['left_shoulder_x'], df['left_shoulder_y'],
                                          df['left_elbow_x'], df['left_elbow_y'])
    return df

if __name__ == '__main__':
    pass

    