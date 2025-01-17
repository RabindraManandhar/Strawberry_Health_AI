from ultralytics import YOLO

from src.utils import *


def image_capture_track_transform_classify(model_path, camera_index, saved_frames_dir):

    # Load the configuration from config.yaml
    config = load_config_with_env()

    # Get the class names from the config
    class_names = config["class_names"]

    # Load the trained model
    model = YOLO(model_path)

    results_list = []

    # Ensure the directory to save frame exists and is empty
    os.makedirs(saved_frames_dir, exist_ok=True)
    clear_folder(saved_frames_dir)

    # Open a connection to the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit the video stream.")

    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture frame.")
            break

        # Apply image transformations to the frame before classification
        transformed_frame = apply_image_transformations(frame)

        # Resize to YOLO's input resolution (e.g., 640x640)
        resized_image = cv2.resize(transformed_frame, (640, 640))

        # Run YOLOv8 object detection and tracking on the frame
        results = model.track(resized_image, persist=True)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot() if len(results) > 0 and hasattr(results[0], 'plot') else frame
        annotated_frame = results[0].plot()

        # Convert the annotated frame to grayscale
        # gray_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)

        # Display the annotated frame
        cv2.imshow("Camera" + str(camera_index), annotated_frame)

        # Save the current frame as an image file
        frame_filename = f"Frame_{frame_count + 1}.jpg"
        frame_filepath = os.path.join(saved_frames_dir, frame_filename)
        cv2.imwrite(frame_filepath, annotated_frame)  # Save the frame as an image
        print(f"Saved frame: {frame_filepath}")

        # check if any class (disease) is detected
        predictions = (
            results[0].boxes.cls.cpu().numpy()
        )  # Get class indices of the predictions

        if len(predictions) == 0:
            # No predictions (no disease detected), classify as 'Healthy'
            results_list.append(
                {
                    "image": frame_filename,  # Save the frame filename
                    "classification": "Healthy",
                    "detected_classes": [],
                }
            )
        else:
            # One or more diseases detected, classify as 'Unhealthy'
            detected_classes = [class_names[int(idx)] for idx in predictions]
            results_list.append(
                {
                    "image": frame_filename,  # Save the frame filename
                    "classification": "Not Healthy",
                    "detected_classes": detected_classes,
                }
            )

        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display windows
    cap.release()
    cv2.destroyAllWindows()

    # Save all the results together
    output_file_path = "classification_results.txt"  # Define your output file path
    save_classification_results(results_list, output_file_path)
