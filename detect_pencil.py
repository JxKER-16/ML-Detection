# detect_pencil.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model("pencil_classifier.h5")

# Define model input size (must match training parameters)
img_height, img_width = 64, 64

def preprocess_frame(frame):
    """
    Resize and normalize the frame to prepare for model prediction.
    """
    resized = cv2.resize(frame, (img_width, img_height))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)

def generate_relation_map(frame):
    """
    Generate a simple black & white heat map by converting to grayscale,
    applying Gaussian blur, and then thresholding.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    ret, heat_map = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    return heat_map

def main():
    cap = cv2.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Preprocess the frame for prediction
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)[0][0]
        
        # Determine label based on prediction threshold 0.5
        if prediction >= 0.5:
            label = "Pencil"
        else:
            label = "Not a Pencil"
        
        # Overlay the prediction label on the camera feed
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Generate the relation map (heat map)
        heat_map = generate_relation_map(frame)
        
        # Show both the camera feed and relation map
        cv2.imshow("Camera Feed", frame)
        cv2.imshow("Relation Map", heat_map)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
