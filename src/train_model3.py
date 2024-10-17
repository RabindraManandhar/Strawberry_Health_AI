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
        momentum=0.937,  # Momentum
        weight_decay=0.0005,  # Weight decay for regularization
        val=True,
        plots=True,
        # Augmentation
        hsv_h=0.015,  # HSV-Hue augmentation (0.0-1.0)
        hsv_s=0.7,  # HSV-Saturation augmentation (0.0-1.0)
        hsv_v=0.4,  # HSV-Value augmentation (0.0-1.0)
        degrees=0.5, # Rotates the image randomly within the specified degree range,
        translate=0.1, # Translates the image horizontally and vertically
        scale=0.5,  # Image scaling (0.0-2.0)
        shear=0.5,  # Image shear (0.0-2.0)
        perspective=0.01,  # Perspective warp (0.0-0.001)
        flipud=0.5,  # Probability of vertical flip
        fliplr=0.5,  # Probability of horizontal flip
        mosaic=1.0,  # Probability of applying mosaic augmentation (mixes 4 images into one)
        mixup=0.2,  # Probability of applying mixup (blends two images)
        device=(0,1) # multi gpu
    )

    print(f"Model fine-tuned for {epochs} epochs.")