import os
import yaml
from dotenv import load_dotenv

# Load environment variable from the .env file
load_dotenv()

def load_config_with_env(config_file="config/config.yaml"):
    """Loads the YAML config file and substitutes environment variables."""

    # Use absolute path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to src/
    config_file_path = os.path.join(script_dir, '..', config_file)  # Go up one level to find the config folder

    # Check if the file exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"No such file: {config_file_path}")

    # Load the YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Retrieve the dataset download key from environment variables
    key = os.getenv("STRAWBERRY_DISEASE_KEY")

    # If the API key is not set, raise an error or handle it
    if key is None:
        raise ValueError("Key not found. Please set the STRAWBERRY_DISEASE_KEY environment variable.")

    # Substitute environment variable in the dataset URL
    config['dataset_url'] = config['dataset_url'].replace("${STRAWBERRY_DISEASE_KEY}", key)

    return config

'''
if __name__ == "__main__":
    config = load_config_with_env()
    print(config)
'''


