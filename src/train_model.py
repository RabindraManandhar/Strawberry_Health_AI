from ultralytics import YOLO

def train_model(data_file, batch_size, epochs, img_size):
    # Load the pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt')
    print("Pre-trained YOLOv8n model loaded successfully.")

    # Fine-tune the model on the training set in the roboflow dataset.
    model.train(
        # Training hyperparameters
        data=data_file,
        batch=batch_size,
        epochs=epochs,
        imgsz=img_size,
        lr0=0.01,
        device='mps' # mac m1 => mps
    )

    print(f"Model fine-tuned for {epochs} epochs.")