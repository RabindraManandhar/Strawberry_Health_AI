from ultralytics import YOLO
import os
import json
import pandas as pd
import glob

def validate_model(model_path, data_file):
    # Load the trained model (after fine-tuning)
    model  = YOLO(model_path)

    # Validate the model on the validation set
    metrics = model.val(data=data_file, save_json=True, device="mps")

    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # a list contains map50-95 of each category

    # Find the most recent validation directory in 'runs/detect/val'
    latest_val_dir = get_latest_run_dir('runs/detect')

    if latest_val_dir:
        # Construct the path to the predictions.json file
        json_file_path = os.path.join(latest_val_dir, "predictions.json")

        # Convert JSON to CSV using pandas
        csv_file_path = os.path.join(latest_val_dir, "predictions.csv")
        if os.path.exists(json_file_path):
            print(f"Converting {json_file_path} to {csv_file_path} using pandas...")
            json_to_csv_pandas(json_file_path, csv_file_path)
        else:
            print(f"{json_file_path} not found, skipping JSON to CSV conversion.")
    else:
        print("No validation results found.")

    return metrics

def get_latest_run_dir(base_dir):

    # List all subdirectories in base_dir
    subdirs = glob.glob(os.path.join(base_dir, '*/'))
    if subdirs:
        # Sort subdirectories by modification time (latest first)
        latest_subdir = max(subdirs, key=os.path.getmtime)
        return latest_subdir
    else:
        return None

def json_to_csv_pandas(json_file_path, csv_file_path):

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Create an empty list to store the data
    rows = []

    # Extract the relevant fields and append to the list
    for prediction in data:
        image_id = prediction["image_id"]
        class_id = prediction["category_id"]
        confidence = prediction["score"]
        bbox = prediction["bbox"]

        # YOLOv8 bbox format: [xmin, ymin, width, height], convert to [xmin, ymin, xmax, ymax]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]  # xmin + width
        ymax = bbox[1] + bbox[3]  # ymin + height

        # Append each row to the list
        rows.append([image_id, class_id, confidence, xmin, ymin, xmax, ymax])

    # Create a DataFrame from the list
    df = pd.DataFrame(rows, columns=["image_id", "class", "confidence", "xmin", "ymin", "xmax", "ymax"])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"CSV file saved as {csv_file_path}.")