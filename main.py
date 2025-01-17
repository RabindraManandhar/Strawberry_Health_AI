import os
import sys

# Add the 'src' directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "src"))

from src.utils import *
from src.download_data import download_and_extract_dataset

from src.train_model import train_model
from src.download_images_from_server import (
    list_images,
    download_image,
    download_all_images,
)

from src.image_capture_track_transform_classify import (
    image_capture_track_transform_classify,
)
from src.image_transform_classify import image_transform_and_classify

if __name__ == "__main__":
    # load configuration
    config = load_config_with_env()

    # Download dataset if necessary
    # Extract parameters from the config
    dataset_url = config["dataset_url_1"]
    data_dir = config["data_dir"]
    download_and_extract_dataset(dataset_url, data_dir)

    # Fine-tuning the preloaded model on roboflow dataset with Transfer learning
    # Extract parameters from the config
    data_file = os.path.join(config["base_dir"], "data", "data.yaml")
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    img_size = config["img_size"]
    train_model(data_file, batch_size, epochs, img_size)

    """
    # Downloading images from server
    server_url = config["server_url"]
    download_dir = config["download_dir"]
    list_images(server_url)  # listing all the images stored in server
    download_image(
        "abc.jpg", server_url, download_dir
    )  # downloading a specific image from the server
    download_all_images(
        server_url, download_dir
    )  # downloading all the images stored in the server
    """

    model_path = "runs/detect/train7/weights/last.pt"

    # Image capture, transform, tracking and classify
    camera_index = 0
    saved_frames_dir = config["saved_frames_dir"]
    image_capture_track_transform_classify(model_path, camera_index, saved_frames_dir)

    # Image Transform and classify
    input_images_dir = config["input_images_dir"]
    transformed_image_dir = config["transformed_images_dir"]
    image_transform_and_classify(model_path, input_images_dir, transformed_image_dir)
