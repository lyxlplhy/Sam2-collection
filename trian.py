from ultralytics import YOLO


model = YOLO(r"D:\sam2\ultralytics-main\ultralytics-main\yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r"D:\sam2\ultralytics-main\ultralytics-main\data3\conver.yaml", epochs=100, imgsz=640)