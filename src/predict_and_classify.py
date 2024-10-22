import os
from ultralytics import YOLO
from src.utils import load_config_with_env

def predict_and_classify(model_path, new_image_path):

    # Load the configuration from config.yaml
    config = load_config_with_env()

    # Get the class names from the config
    class_names = config['class_names']

    # Load the trained model
    model  = YOLO(model_path)

    results_list = []

    # Loop through all images in the new dataset directory
    for img_file in os.listdir(new_image_path):
        new_img_path = os.path.join(new_image_path, img_file)

        # Check if the file is an image (by extension)
        if img_file.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
            print(f"Running inference on {img_file}...")

            # Run inference on the image
            results = model.predict(source=new_img_path, conf=0.5, save=True, save_txt=True)
            print(f"Finished inference on {new_img_path}")

            # Check if any class (disease) is detected
            predictions = results[0].boxes.cls.cpu().numpy()  # Get class indices of the predictions

            if len(predictions) == 0:
                # No predictions (no disease detected), classify as 'Healthy'
                results_list.append({
                    'image': img_file,
                    'classification': 'Healthy',
                    'detected_classes' : []
                })
            else:
                # One or more disease detected, classify as 'Unhealthy'
                detected_classes = [class_names[int(idx)] for idx in predictions]
                results_list.append({
                    'image': img_file,
                    'classification': 'Not Healthy',
                    'detected_classes' : detected_classes
                })

    # Print all the results together
    print_classification_results(results_list)

def print_classification_results(results_list):
    # Prints the classification in rows
    print(f"{'Image':<20} {'Classification':<20} Detected Classes")
    print('-' * 70)

    # Print each image result
    for result in results_list:
        detected_classes_str = ', '.join(result['detected_classes']) if result['detected_classes'] else 'None'
        print(f"{result['image']:<20} {result['classification']:<20} {detected_classes_str}")



