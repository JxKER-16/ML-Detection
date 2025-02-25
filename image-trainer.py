import cv2
import os
import time
import json
import base64
from datetime import datetime

def create_folder(folder_name):
    """Creates a folder if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_image_json(image_path, json_path, label, serial):
    """Reads an image, encodes it in base64, and saves metadata in JSON."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    data = {
        "label": label,
        "serial": serial,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_data": encoded_string
    }

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)

def main():
    label = input("Enter label prefix for images (e.g., 'goods'): ").strip()
    
    # Create folders to store images and JSON files
    images_folder = "captured_images"
    json_folder = "json_images"
    create_folder(images_folder)
    create_folder(json_folder)

    cap = cv2.VideoCapture(0)  # Open default camera

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    serial = 1
    print("Press 'q' to stop capturing images.")

    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Exiting.")
            break

        # Display the camera feed
        cv2.imshow("Camera Feed - Press 'q' to exit", frame)

        # Capture an image every 2 seconds
        if time.time() - last_capture_time >= 2:
            image_filename = f"{label}_{serial}.jpg"
            image_path = os.path.join(images_folder, image_filename)

            # Save the image
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")

            # Convert to JSON
            json_filename = f"{label}_{serial}.json"
            json_path = os.path.join(json_folder, json_filename)
            save_image_json(image_path, json_path, label, serial)
            print(f"JSON saved: {json_path}")

            serial += 1
            last_capture_time = time.time()  # Reset timer

        # Press 'q' to exit the loop and close camera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
