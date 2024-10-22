import cv2
from ultralytics import YOLO

from src.utils import load_config_with_env

def capture_and_track(model_path, camera_index):

    # Load the configuration from config.yaml
    config = load_config_with_env()

    # Get the class names from the config
    class_names = config['class_names']

    # Load the trained model
    model = YOLO(model_path)

    results_list = []

    # Open a connection to the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit the video stream.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture frame.")
            break

        # Run YOLOv8 object detection and tracking on the frame
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Camera" + str(camera_index), annotated_frame)

        # check if any class (disease) is detected
        predictions = results[0].boxes.cls.cpu().numpy()  # Get class indices of the predictions

        if len(predictions) == 0:
            # No predictions (no disease detected), classify as 'Healthy'
            results_list.append({
                'image': f"Frame_{len(results_list) + 1}",
                'classification': 'Healthy',
                'detected_classes': []
            })
        else:
            # One or more diseases detected, classify as 'Unhealthy'
            detected_classes = [class_names[int(idx)] for idx in predictions]
            results_list.append({
                'image': f"Frame_{len(results_list) + 1}",
                'classification': 'Not Healthy',
                'detected_classes': detected_classes
            })

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display windows
    cap.release()
    cv2.destroyAllWindows()

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