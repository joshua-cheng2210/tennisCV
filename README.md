# project purpose
- this is my intorductory project to experience computer vision and deep learning

# library used
- YoloV5: to detect objects
- opencv

# requirements
- only works for procesing tennis videos that are recorded from behind the players. (it doesn't work for side views)
- for better accuracy, the video should be recorded from a fixed position and the camera should not move
- the video should be shot in 24 fps

# Learnings
- some pre trained models are not accurate and specialized for your purpose of your project, so you might need to train your own model
- when training your own model weights, you need to have a lots of data
    - got data to train to detect the tennis ball from roboflow at https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6
    - got the data to train the tennis court from https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link 
- using kaggle's GPU cause my computer only has CPU and it is slow
- the differences betwwen the GPUs offered by kaggle. T4 or P100.
- learn the differences here https://www.kaggle.com/discussions/getting-started/561774
- get some introduction to some libraries like torch, torchvision, opencv to train tennis court keypoint detection. (TODO: check more how to use them)
- a new method of computer vision called keypoint detection, which is used to detect the keypoints of the tennis court
- after getting all the models you need, you can finally proceed to evaluating your gameplay
- had some troubles with git pushing stuff when i tried to upload the input images and videos. (solutions: remove the .git from the current working folder --> git clone from the repo again --> move the .git file from that newly cloned repo to working folder -> should be back working)
- how to fill in missing Rows in a pandas table. use .interpolate() or .ffill() or .bfill() methods
- use cv2 moduel to easily edit each frame of the video
- and then since you know the length measurements of the court, you can calculate the positions of the players o the ball relative to the court. and lastly, you can do further analysis or calculations on the game play like ball speed, player speed, player distance from the ball, etc.

# topics to look up
- fully connected neural networks: each neuron in one layer is connected to every neuron in the next layer
- other types of layers in neural networks: 
  - convolutional layers
  - pooling layers
  - recurrent layers
- linear transformations in neural networks
- importance of non-linear activation functions
- universal approximation theorem

# further improvements
- reduce the number of requirements needed to run this
- scale it to predict
- try to make it work for real time analysis - even though the script is built to process the full video

# some resume bulletpoints (GPT generated)
- Developed a computer vision pipeline using YOLOv5 and OpenCV to detect tennis balls and court lines in match videos, enabling objective post-game analysis.
- Fine-tuned object detection and keypoint models with custom datasets sourced from Roboflow and other repositories, learning to handle domain-specific training challenges and data acquisition.
- Leveraged Kaggle-provided GPUs (T4 and P100) to accelerate deep learning model training, and gained hands-on experience with PyTorch, torchvision, and large-scale dataset management.
- Automated frame-by-frame video processing and data extraction with OpenCV and pandas, applying techniques such as interpolation and data imputation to handle missing data.
- Engineered a system to compute precise player and ball positions relative to the tennis court, laying the foundation for advanced sports analytics and gameplay evaluation.
- Troubleshot and resolved practical issues in version control (git) and large data uploads, improving workflow efficiency for machine learning experiments.