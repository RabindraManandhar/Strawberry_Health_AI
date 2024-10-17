import os
from ultralytics import YOLO

def predict_and_classify(model_path, new_image_path):

    # Load the trained model
    model  = YOLO(model_path)

    # Define the class names (disease names)
    class_names = ['Angular Leafspot', 'Anthracnose Fruit Rot', 'Blossom Blight', 'Gray Mold', 'Leaf Spot',
                   'Powdery Mildew Fruit', 'Powdery Mildew Leaf']

    # Loop through all images in the new dataset directory
    for img_file in os.listdir(new_image_path):
        new_img_path = os.path.join(new_image_path, img_file)

        # Check if the file is an image (by extension)
        if img_file.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
            print(f"Running inference on {img_file}...")

            # Run inference on the image
            results = model.predict(source=new_img_path, save=True, save_txt=True)
            print(f"Finished inference on {new_img_path}")

            # Check if any class (disease) is detected
            class_indices = results[0].boxes.cls.cpu().numpy() # Get class indices of the predictions

            if len(class_indices) == 0:
                # No predictions (no disease detected), classify as 'Healthy'
                print(f"Image {img_file} is classified as 'Healthy'.")
            else:
                # One or more disease detected, classify as 'Unhealthy'
                detected_classes = [class_names[int(idx)] for idx in class_indices]
                print(f"Image {img_file} is classified as 'Unhealthy'. Detected disease: {detected_classes}.")


