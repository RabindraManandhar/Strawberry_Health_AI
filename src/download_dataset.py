# Downloading and extracting dataset with original images
import os
import zipfile
import requests


# Downloading dataset
def download_and_extract_dataset(dataset_url, data_dir):

    # Make directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if the dataset directory is empty
    if not os.listdir(data_dir):
        print(f"Dataset not found. Downloading from {dataset_url}")

        # Download the dataset
        response = requests.get(dataset_url)

        # Save the response content as a zip file
        zip_path = os.path.join(data_dir, "dataset.zip")
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Check if the file is a valid zip file before extracting
        if zipfile.is_zipfile(zip_path):
            # Unzip the dataset
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Dataset successfully downloaded and extracted to {data_dir}")

            # Delete the .zip file after extraction
            os.remove(zip_path)
            print(f"Zip file {zip_path} has been deleted.")

            return True
        else:
            print("Error: The downloaded file is not a valid ZIP file.")
            return False
    else:
        print(f"Dataset already exists at {data_dir}.")


"""
# Downloading dataset and extracting it in grayscale images(if color originally)

import os
import zipfile
import requests
import cv2  # OpenCV for image processing
import numpy as np  # To handle image data in OpenCV

def download_and_extract_dataset(dataset_url, data_dir):

    # Make directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if the dataset directory is empty
    if not os.listdir(data_dir):
        print(f"Dataset not found. Downloading from {dataset_url}...")

        # Download the dataset
        response = requests.get(dataset_url)

        # Save the response content as a zip file
        zip_path = os.path.join(data_dir, "dataset.zip")
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Check if the file is a valid zip file before extracting
        if zipfile.is_zipfile(zip_path):
            # Unzip the dataset
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Iterate over the contents of the zip file
                for file_info in zip_ref.infolist():
                    # If the current item is a directory, create the directory
                    if file_info.is_dir():
                        os.makedirs(os.path.join(data_dir, file_info.filename), exist_ok=True)
                        print(f"Created directory: {file_info.filename}")
                    else:
                        # Handle image files (you can customize this for specific image formats if needed)
                        if file_info.filename.endswith(('.png', '.jpg', '.jpeg')):
                            # Open the image file as binary data
                            with zip_ref.open(file_info) as file:
                                # Read the image from the binary file
                                img_data = np.frombuffer(file.read(), np.uint8)
                                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # Decode the image

                                # Convert the image to grayscale using OpenCV
                                grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                                # Define the path where the grayscale image will be saved
                                output_img_path = os.path.join(data_dir, file_info.filename)

                                # Create directory if it doesn't exist
                                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

                                # Save the grayscale image using OpenCV
                                cv2.imwrite(output_img_path, grayscale_img)
                                print(f"Converted {file_info.filename} to grayscale and saved to {output_img_path}")
                        else:
                            # Handle non-image files (e.g., label files)
                            output_file_path = os.path.join(data_dir, file_info.filename)
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                            # Extract non-image files without modification
                            with zip_ref.open(file_info) as src_file, open(output_file_path, 'wb') as dst_file:
                                dst_file.write(src_file.read())
                                print(f"Extracted non-image file: {file_info.filename}")

            print(f"Dataset successfully downloaded and extracted to {data_dir}")

            # Delete the .zip file after extraction
            os.remove(zip_path)
            print(f"Zip file {zip_path} has been deleted.")

            return True
        else:
            print("Error: The downloaded file is not a valid ZIP file.")
            return False
    else:
        print(f"Dataset already exists at {data_dir}.")
"""
