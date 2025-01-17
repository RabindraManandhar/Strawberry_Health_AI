from ultralytics import YOLO


def train_model(data_file, batch_size, epochs, img_size):
    # Load the pre-trained YOLOv8n model
    model = YOLO("yolov8n-seg.pt")
    print("Pre-trained YOLOv11n model loaded successfully.")

    # Fine-tune the model on the training set in the roboflow dataset.
    model.train(
        # Training hyperparameters
        data=data_file,
        batch=batch_size,
        epochs=epochs,
        imgsz=img_size,
        patience=10,  # number of epochs to wait without improvement in validation metrics before early stopping
        device=0,  # single gpu
        optimizer="AdamW",  # AdamW due to its ability to handle complex models like object detection and image classification.
        lr0=0.001,  # Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
        momentum=0.937,  # beta1 for Adam optimizers
        weight_decay=0.001,  # In AdamW weight_decay helps prevent overfitting, which is important when you want the model to generalize well, especially in disease classification where features could be subtle.
        box=7.5,  # Weight of box loss for better IoU and mAP
        cls=0.5,  # Weight of classification loss for better precision
        dfl=1.5,  # Weight of distribution focal loss for fine-grained classification.
        plots=True,
        # Augmentation
        hsv_h=0.02,  # HSV-Hue augmentation (0.0-1.0)
        hsv_s=0.7,  # HSV-Saturation augmentation (0.0-1.0)
        hsv_v=0.4,  # HSV-Value augmentation (0.0-1.0)
        degrees=10,  # Rotates the image randomly within the specified degree range,
        translate=0.1,  # Translates the image horizontally and vertically
        scale=0.3,  # Image scaling
        shear=0.2,  # Image shear
        perspective=0.01,  # Perspective warp
        flipud=0.1,  # Probability of vertical flip
        fliplr=0.5,  # Probability of horizontal flip
        mosaic=1.0,  # Probability of applying mosaic augmentation (mixes 4 images into one)
        mixup=0.2,  # Probability of applying mixup (blends two images)
    )

    print(f"Model fine-tuned for {epochs} epochs.")
