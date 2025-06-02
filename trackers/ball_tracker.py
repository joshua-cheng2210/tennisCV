from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd
class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.interpolated_positions = []

    def interpolate_ball_positions(self, ball_positions, read_from_stub=False, stub_path=None):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2']) # ['x1','y1','x2','y2'] is not 2 coords. its the bounding box coordinates

        self.interpolated_positions = df_ball_positions.isnull().any(axis=1)

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.ffill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        # for i, x in enumerate(df_ball_positions.to_numpy().tolist()):
        #     print(f"frame {i}: iterpolated = {self.interpolated_positions[i]}: {x}")

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_positions, f)
            except:
                pass

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 22
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

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

    def detect_frame(self,frame, _conf=0.15):
        results = self.model.predict(frame, conf=_conf)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self,video_frames, ball_detections):
        output_video_frames = []
        index = 0
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                if self.interpolated_positions[index]:
                    color = (0, 165, 255) # bgr format
                    cv2.putText(frame, f"Ball ID: {track_id}; interpolated: 1",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                else:
                    color = (0, 255, 255)
                    cv2.putText(frame, f"Ball ID: {track_id}: interpolated: 0",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            output_video_frames.append(frame)
            index += 1
        
        return output_video_frames


    