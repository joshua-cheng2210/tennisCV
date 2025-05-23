# !dir tennis-ball-detection-6
from ultralytics import YOLO
# data_yaml_path = fr"{dataset.location}\data.yaml"
data_yaml_path = fr"tennis-ball-detection-6\data.yaml"
print("data_yaml_path --> ", data_yaml_path)
model = YOLO("yolov8x.pt")

# Train the model
model.train(data=data_yaml_path, epochs=120, imgsz=640)
