import cv2
import moviepy
from tqdm import tqdm
# Reads a video file and returns its frames as a list of images
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Save a list of video frames to a file
def save_video(output_video_frames, output_video_path, mp4=False):
    print("saving video")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in tqdm(output_video_frames, desc="Saving video"):
        out.write(frame)
    out.release()

    if mp4:
        convert_avi_to_mp4(output_video_path, output_video_path.replace('.avi', '.mp4'))

def convert_avi_to_mp4(avi_path, mp4_path):
    try:
        clip = moviepy.VideoFileClip(avi_path)
        clip.write_videofile(mp4_path, codec="libx264")
        clip.close()
        return True
    except Exception as e:
        print(f"Error converting {avi_path} to {mp4_path}: {e}")
        return False