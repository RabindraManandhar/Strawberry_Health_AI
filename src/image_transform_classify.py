from ultralytics import YOLO

from src.utils import *


def image_transform_and_classify(model_path, input_images_dir, transformed_images_dir):

    # Load the configuration from config.yaml
    config = load_config_with_env()

    # Get the class names from the config
    class_names = config["class_names"]

    # Load the trained model
    model = YOLO(model_path)

    results_list = []

    # Ensure the transformed_dir exists
    os.makedirs(transformed_images_dir, exist_ok=True)
    clear_folder(transformed_images_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_path = os.path.join(input_images_dir, filename)

            # Load image
            image = cv2.imread(image_path)

            if image is None:
                print("Error: Could not read the image. Please check the file path.")
                continue

            # Apply the image transformations
            transformed_image = apply_image_transformations(image)

            # Resize to YOLO's input resolution (e.g., 640x640)
            resized_image = cv2.resize(transformed_image, (640, 640))

            # Run YOLOv8 object detection and tracking on the frame
            results = model.predict(resized_image)

            # Visualize the results on the frame
            annotated_image = results[0].plot()

            # Convert the annotated frame to grayscale
            # gray_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)

            # Display the annotated frame
            cv2.imshow("Image", annotated_image)

            # Save the transformed image in the transformed_dir
            transformed_image_filename = f"T_{filename}"
            transformed_filepath = os.path.join(
                transformed_images_dir, transformed_image_filename
            )
            cv2.imwrite(
                transformed_filepath, annotated_image
            )  # Save the frame as an image
            print(f"Transformed image: {transformed_filepath}")

            # check if any class (disease) is detected
            predictions = (
                results[0].boxes.cls.cpu().numpy()
            )  # Get class indices of the predictions

            if len(predictions) == 0:
                # No predictions (no disease detected), classify as 'Healthy'
                results_list.append(
                    {
                        "image": transformed_image_filename,  # Save the frame filename
                        "classification": "Healthy",
                        "detected_classes": [],
                    }
                )
            else:
                # One or more diseases detected, classify as 'Unhealthy'
                detected_classes = [class_names[int(idx)] for idx in predictions]
                results_list.append(
                    {
                        "image": transformed_image_filename,  # Save the frame filename
                        "classification": "Not Healthy",
                        "detected_classes": detected_classes,
                    }
                )

            # Save all the results together
            output_file_path = (
                "classification_results.txt"  # Define your output file path
            )
            save_classification_results(results_list, output_file_path)
