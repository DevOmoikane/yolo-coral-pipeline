from ultralytics import YOLO

model = YOLO('weights/best.pt')  
model.export(format='tflite', imgsz=512)  # Adjust imgsz based on your trained model's image size
