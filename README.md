# drone_project
This project is technically still in work, with more and better training data to increase the reliability of the gesture control. A single camera makes it difficult to infer differences between similiarly appearing movements, so a second camera view in the future could help the model immensely. 

Set up:
installed ultralytics then torch, then DJITelloPy from https://github.com/damiafuentes/DJITelloPy/blob/master/README.md 

the main flight control code is located at flight_control.py, which needs two parts to work. 

1. First, DJITelloPy must be installed and the drone first should be connected throught the DJI Tello App to initialize it. 
2. Then, the MLP model and weights must available. The training process was done first using frame_extractor.py to split a video of demo gestures into individual frames, with a frames per second you can adjust in the code. Then, I manually labeled the inflection points of the gesture changes and their corresponding velocity values from [-1,1]. Afterwards, the data_label.py will go through the frames and interpolate between the manually labeled inflection points to produce a training data df. control_net_train.py can be used to train the MLP

The camera is positioned directly infront of the individual, capturing from head to knees with some additional margin for better results. The camera used was a Samsung S23 Ultra connected to PC via USB through the Iriun webcam app. I noticed that using other camera setups yields different results, meaning the model is likely dependent on the specific camera (FOV, lens curve, etc).

the following gesture controls are: 
takeoff: left arm raises
land: both arms raised

throttle: left arm up and down
yaw: left arm left and right while elbow is at fixed position (cannot combine throttle and yaw)
pitch: right arm front and back (the elbow is held up and in front, moving only then arm moves back for - pitch)
roll: right arm left and right while elbow is at fixed position

