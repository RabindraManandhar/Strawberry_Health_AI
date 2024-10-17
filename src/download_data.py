import os
import zipfile
import requests

def download_and_extract_dataset(dataset_url, output_dir):

    # Make directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the dataset directory is empty
    if not os.listdir(output_dir):
        print(f"Dataset not found. Downloading from {dataset_url}")

        # Download the dataset
        response = requests.get(dataset_url)

        # Save the response content as a zip file
        zip_path = os.path.join(output_dir, "dataset.zip")
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Check if the file is a valid zip file before extracting
        if zipfile.is_zipfile(zip_path):
            # Unzip the dataset
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"Dataset successfully downloaded and extracted to {output_dir}")

            # Delete the .zip file after extraction
            os.remove(zip_path)
            print(f"Zip file {zip_path} has been deleted.")

            return True
        else:
            print("Error: The downloaded file is not a valid ZIP file.")
            return False
    else:
        print(f"Dataset already exists at {output_dir}.")



