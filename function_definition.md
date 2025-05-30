This file summarizes all functions in each .py file in the workspace with brief descriptions.

main.py
--------
main():
  Orchestrates the video analysis pipeline. Reads an input video, initializes player and ball trackers, and (optionally) runs detection, court keypoint detection, player selection, and other analysis steps.

utils/video_utils.py
--------------------
read_video(video_path):
  Reads a video file and returns its frames as a list of images using OpenCV.
save_video(output_video_frames, output_video_path):
  Saves a list of video frames to a file using OpenCV.

trackers/player_tracker.py
--------------------------
PlayerTracker.__init__(self, model_path):
  Initializes the player tracker with a YOLO model.
PlayerTracker.choose_and_filter_players(self, court_keypoints, player_detections):
  Selects and filters player detections based on proximity to court keypoints.
PlayerTracker.choose_players(self, court_keypoints, player_dict):
  Chooses the two players closest to the court keypoints.
PlayerTracker.detect_frames(self, frames, read_from_stub=False, stub_path=None):
  Detects players in each frame, optionally reading/writing results from/to a stub file.
PlayerTracker.detect_frame(self, frame):
  Detects players in a single frame and returns their bounding boxes.
PlayerTracker.draw_bboxes(self, video_frames, player_detections):
  Draws bounding boxes and player IDs on video frames.

utils/__init__.py and trackers/__init__.py
------------------------------------------
These files import functions/classes from their respective modules for easier access.
