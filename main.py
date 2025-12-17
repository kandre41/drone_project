from ultralytics import YOLO
import cv2

# --- Configuration ---
# 1. Load the YOLO model (Replace 'yolo11n.pt' with your specific model file path)
# Use the path to your trained model weights or a pre-trained model like 'yolo11n.pt' for a general model.
path="W:\VSCode\drone_project"
#model = YOLO(f'{path}\weights\\yolo11x-pose.engine') 
model = YOLO(f'{path}\weights\\yolo11l_hand_300.pt')
# 2. Set the Webcam Source
# '0' typically refers to the default primary camera (the built-in laptop camera or a single connected USB camera).
# Use '1', '2', etc., if you have multiple cameras.
WEBCAM_SOURCE = 0

# --- Start Webcam Stream ---
# VideoCapture object to read frames from the camera
cap = cv2.VideoCapture(WEBCAM_SOURCE)

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened. Press 'q' to exit.")

# --- Live Inference Loop ---
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break


# --- Cleanup ---
cap.release()          # Release the webcam capture object
cv2.destroyAllWindows()  # Close all OpenCV display windows
del results
del model