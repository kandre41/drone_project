import cv2
import time
import os
from utils import center_crop
import concurrent.futures

base_path = r"W:\VSCode\drone_project\datasets"
video_name = 'demo5'
video_path = os.path.join(base_path, "videos", f"{video_name}.mp4")
output_dir = os.path.join(base_path, "images", video_name)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

source_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = 8
frame_interval = int(source_fps / target_fps)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0
saved_count = 0
start_time = time.time()

# We use a ThreadPoolExecutor to handle writing to disk asynchronously
# max_workers=2 is usually sufficient for disk I/O
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        grabbed = cap.grab()#grabs the next frame (moves forward)
        if not grabbed:
            break #break when frames run out
        if frame_count % frame_interval == 0:
            success, frame = cap.retrieve()
            if success:
                # CPU Bound: Crop
                frame = center_crop(frame, (640, 640))
                
                # I/O Bound: Save to disk in background thread
                save_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04}.jpg")
                executor.submit(cv2.imwrite, save_path, frame)
    
                saved_count += 1

        frame_count += 1

end_time = time.time()
total_duration = end_time - start_time
cap.release()

print(f"Extracted {saved_count} frames")
print(f"Total time: {total_duration:.2f} seconds")
if saved_count > 0:
    print(f"Speed: {1000 * total_duration / saved_count:.2f} ms/image")