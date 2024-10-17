import os
import sys

# Add the 'src' directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'src'))

from src.utils import load_config_with_env
from src.download_data import download_and_extract_dataset
from src.download_grayscale_data import download_and_extract_grayscale_dataset
from src.train_model2 import train_model
#from src.validate_model import validate_model
from src.predict_and_classify import predict_and_classify

if __name__ == '__main__':
    # load configuration
    config = load_config_with_env()

    # Download dataset if necessary
    # Extract parameters from the config
    dataset_url = config['dataset_url']
    output_dir = config['output_dir']
    #download_and_extract_dataset(dataset_url, output_dir)
    download_and_extract_grayscale_dataset(dataset_url, output_dir)

    # Fine-tuning the preloaded model on roboflow dataset with Transfer learning
    # Extract parameters from the config
    #data_file = os.path.join(config['base_dir'], "data", "data.yaml")
    #epochs = config['epochs']
    #batch_size = config['batch_size']
    #img_size = config['img_size']
    #train_model(data_file, batch_size,epochs, img_size)

    # Validate the trained model
    #model_path = "runs/detect/train/weights/best.pt"
    #model_path = "runs/detect/train2/weights/best.pt"
    #validate_model(model_path, data_file)

    # Predict the trained model
    #model_path = "runs/detect/train/weights/best.pt"
    #model_path = "runs/detect/train2/weights/best.pt"
    #new_image_path = config['new_image_path']
    #classification = predict_and_classify(model_path, new_image_path)