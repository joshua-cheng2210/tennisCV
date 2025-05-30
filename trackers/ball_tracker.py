from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
# from utils import measure_distance, get_center_of_bbox

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_balls(self, court_keypoints, ball_detections):
        ball_detections_first_frame = ball_detections[0]
        chosen_ball = self.choose_balls(court_keypoints, ball_detections_first_frame)
        filtered_ball_detections = []
        for ball_dict in ball_detections:
            filtered_ball_dict = {track_id: bbox for track_id, bbox in ball_dict.items() if track_id in chosen_ball}
            filtered_ball_detections.append(filtered_ball_dict)
        return filtered_ball_detections

    def choose_balls(self, court_keypoints, ball_dict):
        distances = []
        for track_id, bbox in ball_dict.items():
            ball_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(ball_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sorrt the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_balls = [distances[0][0], distances[1][0]]
        return chosen_balls

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    ball_detections = pickle.load(f)
                return ball_detections
            except:
                pass

        num_frames = len(frames)
        for i in range(num_frames):
            ball_dict = self.detect_frame(frames[i])
            ball_detections.append(ball_dict)
            if (i+1) % 10 == 0 or i == num_frames - 1:
                print(f"Processed {i}/{num_frames} frames for ball detection.")

        # for frame in frames:
        #     ball_dict = self.detect_frame(frame)
        #     ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self,video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    