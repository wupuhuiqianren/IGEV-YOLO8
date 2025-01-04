from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('YOLO/yolov8x.pt')

# Export the model （set your parameters, refer to ultralytics）
model.export(format="engine", opset=12, simplify=True, dynamic=True, int8=True,imgsz=640,half=True)
