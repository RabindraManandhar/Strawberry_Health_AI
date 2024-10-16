import os
from ultralytics import YOLO

def run_inference(model_path, new_image_path):

    # Load the trained model
    model  = YOLO(model_path)

    # Loop through all images in the new dataset directory
    for img_file in os.listdir(new_image_path):
        new_img_path = os.path.join(new_image_path, img_file)

        # Check if the file is an image (by extension)
        if img_file.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
            print(f"Running inference on {img_file}...")

            # Run inference on the image
            #results = model(new_img_path)
            results = model.predict(new_img_path, save=True, save_txt=True)
            print(f"Finished inference on {new_img_path}")
