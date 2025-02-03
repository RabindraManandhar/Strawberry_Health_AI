import os
import shutil

import yaml
from dotenv import load_dotenv
import cv2
import numpy as np


# Load environment variable from the .env file
load_dotenv()


def load_config_with_env(config_file="config/config.yaml"):
    """Loads the YAML config file and substitutes environment variables."""

    # Use absolute path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to src/
    config_file_path = os.path.join(
        script_dir, "..", config_file
    )  # Go up one level to find the config folder

    # Check if the file exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"No such file: {config_file_path}")

    # Load the YAML file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Retrieve the dataset download key from environment variables
    key = os.getenv("STRAWBERRY_DISEASE_KEY")

    # Retrieve server url from environment variables
    config["server"]["url"] = os.getenv("SERVER_URL")

    # If the API key is not set, raise an error or handle it
    if key is None:
        raise ValueError(
            "Key not found. Please set the STRAWBERRY_DISEASE_KEY environment variable."
        )

    # Substitute environment variable in the dataset URL
    config["dataset_url"] = config["dataset_url"].replace(
        "${STRAWBERRY_DISEASE_KEY}", key
    )
    return config


# Function to clear the folder
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
            except Exception as e:
                print(f"Error: Could not delete file {file_path}.")


# Zoom image
def zoom_frame(frame, zoom_factor=1.5):
    # Get the dimensions of the image
    height, width = frame.shape[:2]

    # Calculate the cropping box for zoom
    crop_h, crop_w = int(height / zoom_factor), int(width / zoom_factor)
    start_y, start_x = (height - crop_h) // 2, (width - crop_w) // 2

    # Crop and resize in one step
    zoomed_frame = cv2.resize(
        frame[start_y : start_y + crop_h, start_x : start_x + crop_w],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    return zoomed_frame


# Transormation -> Extracting masks and converting LAB (Light, A (red-green) and B (blue-yellow)) channels
def apply_image_transformations(frame):
    # Ensure the frame is a valid numpy array
    if frame is None or not isinstance(frame, (np.ndarray, np.generic)):
        raise ValueError("Invalid frame provided for transformation.")

    # Save the original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Apply zoom to the image
    zoomed_frame = zoom_frame(frame, zoom_factor=1.5)

    # Convert the frame BGR -> HSV color space
    hsv_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for warm colors (pink/orange tones) and create mask
    warm_lower = np.array([136.5, 50, 50])
    warm_upper = np.array([163.5, 255, 255])
    warm_mask = cv2.inRange(hsv_frame, warm_lower, warm_upper)

    # Define HSV color ranges for natural colors and create their masks
    # For strawberries
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([8, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_and(red_mask1, red_mask2)

    # For leaves
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # For petals
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)

    # For stem and branches
    brown_lower = np.array([10, 50, 50])
    brown_upper = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv_frame, brown_lower, brown_upper)

    # Combine natural color masks
    natural_mask = cv2.bitwise_or(red_mask, green_mask)
    natural_mask = cv2.bitwise_or(natural_mask, white_mask)
    natural_mask = cv2.bitwise_or(natural_mask, brown_mask)

    # Invert the mask to get non-target areas
    inverse_mask = cv2.bitwise_not(natural_mask)

    # Invert the natural color mask to target only warm areas
    target_mask = cv2.bitwise_and(warm_mask, inverse_mask)

    # Apply white balance to the entire image
    # Convert frame back HSV -> BGR color space and then BGR -> LAB color space
    bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    lab_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)

    # Correct only the non-target areas
    l, a, b = cv2.split(lab_frame)

    # Reduce A (red-green) and B (blue-yellow) channels in warm areas
    a = cv2.subtract(a, np.where(target_mask > 0, 75, 0).astype(np.uint8))
    b = cv2.subtract(b, np.where(target_mask > 0, 10, 0).astype(np.uint8))

    # Boost L (luminance) in affected areas to reveal natural colors
    l = cv2.add(l, np.where(target_mask > 0, 255, 0).astype(np.uint8))

    # Merge the LAB Channels back
    corrected_lab = cv2.merge((l, a, b))

    # Convert frame back LAB -> BGR color space
    transformed_frame = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return transformed_frame


# Transformation -> Extracting masks and converting HSV (Hue, Saturation, Brightness) values
"""
def apply_image_transformations(frame):
    #Ensure the frame is a valid numpy array
    if frame is None or not isinstance(frame, (np.ndarray, np.generic)):
        raise ValueError("Invalid frame provided for transformation.")

    # Save the original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Apply zoom to the image
    zoomed_image = zoom_image(frame, zoom_factor=1.5)

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2HSV)

    # Define the range for purple/purple tones in HSV
    lower_purple = np.array([136.5, 50, 50])  # Lower bound of purple/magenta
    upper_purple = np.array([163.5, 255, 255])  # Upper bound of purple/magenta

    # Create a mask for purple/purple colors
    purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)

    # Refine the mask by applying morphological operations
    kernel = np.ones((5, 5), np.uint8)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

    # Extract HSV channels
    h, s, v = cv2.split(hsv_frame)

    # Replace purple hues with green
    h[purple_mask > 0] = 75 # Set hue to spring green for leaves
    s[purple_mask > 0] = np.minimum(s[purple_mask > 0] - 50, 255) # Max out saturation for green
    #v[purple_mask > 0] = np.minimum(s[purple_mask > 0], 255) # Increase brightness

    # Ensure the red hues for strawberries are preserved
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_and(red_mask1, red_mask2)

    # Preserve the red areas
    h[red_mask > 0] = hsv_frame[:, :, 0][red_mask > 0]
    s[red_mask > 0] = hsv_frame[:, :, 1][red_mask > 0]
    v[red_mask > 0] = hsv_frame[:, :, 2][red_mask > 0]

    # Merge the adjusted channels back into an HSV image
    enhanced_hsv = cv2.merge([h, s, v])

    # Convert back to BGR color space
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Blend the modified and original images for smoother results
    blended_result = cv2.addWeighted(frame, 0.2, result, 0.8, 0)

    # Resize the transformed frame back to the original dimensions
    transformed_frame = cv2.resize(blended_result, (original_width, original_height))

    return transformed_frame
"""


# Function to save the classification results to a file
def save_classification_results(results_list, output_file_path):

    # Save the classification results to an output file.
    with open(output_file_path, "w") as file:
        # Write the header
        file.write(f"{'Image':<50} {'Classification':<20} Detected Classes\n")
        file.write("-" * 70 + "\n")

        # Write each result to the file
        for result in results_list:
            detected_classes_str = (
                ", ".join(result["detected_classes"])
                if result["detected_classes"]
                else "None"
            )
            file.write(
                f"{result['image']:<50} {result['classification']:<20} {detected_classes_str}\n"
            )

    print(f"Classification results saved to {output_file_path}")


def print_classification_results(results_list):
    # Prints the classification in rows
    print(f"{'Image':<20} {'Classification':<20} Detected Classes")
    print("-" * 70)

    # Print each image result
    for result in results_list:
        detected_classes_str = (
            ", ".join(result["detected_classes"])
            if result["detected_classes"]
            else "None"
        )
        print(
            f"{result['image']:<20} {result['classification']:<20} {detected_classes_str}"
        )
