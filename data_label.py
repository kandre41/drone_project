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

for result in results:
    original_name = os.path.basename(result.path)
    save_path = os.path.join(output_dir, f"processed_{original_name}")
    result.save(filename=save_path)


