# import sys
# sys.path.insert(0, './ultralytics')
from ultralytics import YOLO

## testing if you're libraries have bee succesfully downloaded
# there are different models.
# v8 or additional numbers probably means later versions
# n: Nano, s: Small, m: medium, l: large, x: extra-large --> each of this means more accurate but will take longer to train and you're computer may not handle
model = YOLO("yolov8x")
# result = model.predict("input_videos/image.png", save=True) # you can choose to predict on a video or an image
# result = model.predict("input_videos/input_video.mp4", save=True)
# print("printing results\n\n")
# print(result)

## if there is one thing you'll realize, this model is very bad at predicting the tennis ball
### let's try to train our own model
# the code is in training\tennis_ball_detector_training.ipynb

## now you have the training weights of detecting the tennis ball lets proceed to evaluating it
# choose whichever one that is better
# tennis_ball_model = YOLO("models/best.pt")
# tennis_ball_model = YOLO("models/last.pt")
# result = model.predict("input_videos/image.png", conf=0.2, save=True)
# result = tennis_ball_model.predict("input_videos/input_video.mp4", save=True)

result = model.predict("input_videos/image.png", conf=0.2, save=True)
# result = model.predict("input_videos/input_video.mp4", save=True)

## tracking the an object from one frame in a video to another with YOLO
# model = YOLO("yolov8x")
# result = model.track("input_videos/input_video.mp4", save=True)