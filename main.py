import os
import sys
import torch

# Add the 'src' directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "src"))

from src.utils import *
from src.download_dataset import download_and_extract_dataset
from src.download_images_from_server import download_all_images

from src.train_model import train_model
from src.validate_model import validate_model

from src.camera_image_capture_track_transform_classify import (
    camera_image_capture_transform_classify,
)
from src.server_image_transform_classify import server_image_transform_and_classify


if __name__ == "__main__":
    # load configuration
    config = load_config_with_env()

    # Extract required parameters from the config
    dataset_url = config["dataset_url"]
    data_dir = config["data_dir"]
    data_file = os.path.join(config["base_dir"], "data", "data.yaml")
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    img_size = config["img_size"]
    server_url = config["server_url"]
    download_dir = config["download_dir"]
    transormed_frames_dir = config["transormed_frames_dir"]
    transformed_image_dir = config["transformed_images_dir"]

    # Configure device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    #print(device)

    # Download dataset (if necessary)
    download_and_extract_dataset(dataset_url, data_dir)
    
    # Fine-tuning the preloaded model on roboflow dataset with Transfer learning
    train_model(data_file, batch_size, epochs, img_size, device)


    # Last model obtained from training
    model_path = "runs/detect/train/weights/last.pt"

    # Validate the trained model
    validate_model(model_path, data_file, device)


    # Camera captured image transformation, object detection and classification
    camera_index = 0
    camera_image_capture_transform_classify(
        model_path, camera_index, transormed_frames_dir
    )

    # Downloading images stored in the server
    download_all_images(server_url, download_dir)

    # Serverd downloaded image transformation, object detection and classification
    server_image_transform_and_classify(
        model_path=model_path,
        input_images_dir=download_dir,
        transformed_images_dir=transformed_image_dir,
    )

