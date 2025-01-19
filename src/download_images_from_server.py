import requests
import os
import zipfile
from io import BytesIO


def list_images(server_url):
    """Fetch the list of images from the Raspberry Pi server."""
    try:
        response = requests.get(f"{server_url}/list_images")
        response.raise_for_status()  # Check if the request was successful
        images = response.json()  # Get the list of images

        # Print the list of images
        if images:
            print("Images available on the server:")
            for image in images:
                print(image)
        else:
            print(f"No images available on server: {server_url}")

        return images

    except requests.RequestException as e:
        print(f"Error fetching image list: {e}")
        return []


def download_image(filename, server_url, download_dir):

    # Make directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    """Download an image from the Raspberry Pi server."""
    try:
        response = requests.get(f"{server_url}/get_image/{filename}", stream=True)
        response.raise_for_status()  # Check if the request was successful

        filepath = os.path.join(download_dir, filename)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded: {filename}")

    except requests.RequestException as e:
        print(f"Error downloading image {filename}: {e}")


def download_all_images(server_url, download_dir):
    """Download all images from the server as a zip file and extract them."""

    # Make directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    try:
        # Request the zip file from the server
        response = requests.get(f"{server_url}/get_all_images", stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Use BytesIO to store the zip file content in memory
        zip_file = BytesIO(response.content)

        # Extract the zip file
        with zipfile.ZipFile(zip_file, "r") as zip:
            zip.extractall(download_dir)
            print(f"Downloaded and extracted all images to : {download_dir}")

    except requests.RequestException as e:
        print(f"Error downloading zip file: {e}")
