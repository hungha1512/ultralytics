from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="AGAR_lower_prepared/data.yaml",  # Path to dataset configuration file
    epochs=80,  # Number of training epochs
    imgsz=1024,  # Image size for training
    pretrained=False
)

# Evaluate the model's performance on the validation set
metrics = model.val()
