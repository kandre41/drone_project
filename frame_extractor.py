import cv2
import time
from utils.utils import center_crop
path="W:\\VSCode\\drone_project\\datasets\\videos"
video_name = 'demo1'
cap = cv2.VideoCapture(f'{path}\\{video_name}.mp4')

source_fps = cap.get(cv2.CAP_PROP_FPS)
# Calculate how many frames to skip to get 8 FPS
frame_interval = int(source_fps / 8)

frame_id = 0
saved_count = 0
start_time = time.time()
while cap.isOpened():
    # Set the next frame position to "jump" ahead
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    success, frame = cap.read()
    if not success:
        break
    frame = center_crop(frame,(640,640))
    # Save the frame
    cv2.imwrite(f'W:\\VSCode\\drone_project\\datasets\\images\\{video_name}\\{video_name}_frame_{saved_count:04}.jpg', frame)
    
    # Increment by our interval
    frame_id += frame_interval
    saved_count += 1
end_time = time.time()
total_duration = end_time - start_time

cap.release()
print(f"extracted {saved_count} frames")
print(f"total time was {total_duration:.2f} seconds at {1000*total_duration/saved_count:.2f} ms/image")