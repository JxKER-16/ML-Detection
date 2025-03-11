import cv2
import os
import json
import numpy as np
from datetime import datetime

def create_folder(folder_name):
    """Creates a folder if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

class TriangleOrientationClassifier:
    def __init__(self):
        # Visualization settings
        self.colors = {
            'contour': (0, 255, 0),  # Green
            'centroid': (0, 0, 255),  # Red
            'leftmost': (255, 0, 0),  # Blue
            'rightmost': (255, 255, 0),  # Cyan
            'text': (0, 0, 255)  # Red
        }
    
    def analyze_image(self, image_path):
        """Analyze a triangle image and determine its orientation."""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Could not read image", None
        
        # Create visualization
        visualization = image.copy()
        
        # Get dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a list to track our approaches
        approaches = []
        
        # 1. Enhanced color segmentation for dark colors against dark backgrounds
        # First, let's try using the RGB channels directly to find color differences
        
        # Split the image into its RGB channels
        b, g, r = cv2.split(image)
        
        # Create difference images between channels to highlight the colored triangle
        rg_diff = cv2.absdiff(r, g)
        rb_diff = cv2.absdiff(r, b)
        gb_diff = cv2.absdiff(g, b)
        
        # Combine the differences to enhance the triangle visibility
        color_diff = cv2.max(cv2.max(rg_diff, rb_diff), gb_diff)
        
        # Apply threshold to get a binary image
        _, color_thresh = cv2.threshold(color_diff, 15, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        approaches.append(("Color Difference", color_mask))
        
        # 2. Try Contrast Enhancement
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Apply adaptive thresholding on the enhanced image
        enhanced_thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
        
        enhanced_mask = cv2.morphologyEx(enhanced_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        approaches.append(("Enhanced Contrast", enhanced_mask))
        
        # 3. HSV color space with fine-tuned ranges for colored triangles
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges (including dark blue specifically)
        color_ranges = [
            # Dark blue (like in the example)
            (np.array([100, 50, 20]), np.array([140, 255, 100])),
            # Light blue
            (np.array([100, 50, 100]), np.array([140, 255, 255])),
            # Red (two ranges because red wraps around in HSV)
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([160, 70, 50]), np.array([180, 255, 255])),
            # Green
            (np.array([40, 70, 50]), np.array([80, 255, 255])),
            # Yellow
            (np.array([20, 70, 50]), np.array([35, 255, 255])),
            # Dark colors (including black/near-black triangles)
            (np.array([0, 0, 0]), np.array([180, 255, 30]))
        ]
        
        # Create a combined mask for all color ranges
        combined_color_mask = np.zeros_like(gray)
        
        for lower, upper in color_ranges:
            # Create a mask for the color
            mask = cv2.inRange(hsv, lower, upper)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, mask)
        
        # Clean the combined mask
        combined_color_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_color_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        approaches.append(("HSV Color Range", combined_color_mask))
        
        # 4. Edge detection approach
        # Apply Canny edge detection with multiple threshold values
        edges_low = cv2.Canny(image, 30, 90)
        edges_high = cv2.Canny(image, 80, 200)
        
        # Combine the edge images
        edges_combined = cv2.bitwise_or(edges_low, edges_high)
        dilated_edges = cv2.dilate(edges_combined, kernel, iterations=1)
        
        approaches.append(("Edge Detection", dilated_edges))
        
        # 5. For 3D triangles: try different lighting compensation
        # Apply gamma correction to brighten dark areas
        def adjust_gamma(image, gamma=1.0):
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        
        # Brighten the image to see details in dark areas
        brightened = adjust_gamma(image.copy(), gamma=1.5)
        
        # Apply processing to brightened image
        brightened_gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
        _, bright_thresh = cv2.threshold(brightened_gray, 127, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.morphologyEx(bright_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        approaches.append(("Gamma Correction", bright_mask))
        
        # 6. Adaptive multi-level binary thresholding
        # Try multiple threshold levels to catch different triangle colors
        multi_thresh_mask = np.zeros_like(gray)
        
        threshold_levels = [15, 30, 50, 100, 150, 200]
        for thresh_val in threshold_levels:
            _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            multi_thresh_mask = cv2.bitwise_or(multi_thresh_mask, thresh_img)
        
        multi_thresh_mask = cv2.morphologyEx(multi_thresh_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        multi_thresh_mask = cv2.morphologyEx(multi_thresh_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        approaches.append(("Multi-Threshold", multi_thresh_mask))
        
        # Try all approaches to find a triangle
        successful_approach = None
        
        for approach_name, binary_image in approaches:
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Check each contour for triangular shape
            for contour in contours[:5]:  # Check the 5 largest contours
                # Skip tiny contours
                if cv2.contourArea(contour) < 0.005 * (height * width):
                    continue
                
                # Approximate the contour to get a polygon
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.04 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's a triangle or close to one
                if 3 <= len(approx) <= 5:
                    # Mark which approach was successful
                    successful_approach = approach_name
                    
                    # Draw the contour on the visualization
                    cv2.drawContours(visualization, [approx], 0, self.colors['contour'], 2)
                    
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw centroid
                    cv2.circle(visualization, (cx, cy), 5, self.colors['centroid'], -1)
                    
                    # Find the leftmost and rightmost points
                    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                    
                    # Draw leftmost and rightmost points
                    cv2.circle(visualization, leftmost, 7, self.colors['leftmost'], -1)
                    cv2.circle(visualization, rightmost, 7, self.colors['rightmost'], -1)
                    
                    # Calculate distribution of points
                    left_weight = 0
                    right_weight = 0
                    
                    for point in contour:
                        if point[0][0] < cx:
                            left_weight += 1
                        else:
                            right_weight += 1
                    
                    # Calculate center of mass vs geometric center
                    # Find the bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    geometric_center_x = x + w/2
                    
                    # Determine orientation using multiple methods
                    # Method 1: Compare left/right point distribution
                    distribution_orientation = "left-sided" if left_weight > right_weight else "right-sided"
                    
                    # Method 2: Compare centroid to geometric center
                    centroid_orientation = "left-sided" if cx < geometric_center_x else "right-sided"
                    
                    # Method 3: Calculate skewness
                    # Get x-coordinates of contour points
                    x_coords = [point[0][0] for point in contour]
                    # Calculate mean and standard deviation
                    mean_x = np.mean(x_coords)
                    std_x = np.std(x_coords)
                    # Calculate skewness
                    if std_x > 0:
                        skewness = sum((x - mean_x) ** 3 for x in x_coords) / (len(x_coords) * std_x ** 3)
                        skewness_orientation = "left-sided" if skewness > 0 else "right-sided"
                    else:
                        skewness_orientation = "unknown"
                    
                    # Calculate orientation votes
                    votes = {
                        "left-sided": 0,
                        "right-sided": 0
                    }
                    
                    for orientation in [distribution_orientation, centroid_orientation, skewness_orientation]:
                        if orientation in votes:
                            votes[orientation] += 1
                    
                    # Final orientation by voting
                    if votes["left-sided"] > votes["right-sided"]:
                        final_orientation = "left-sided"
                    elif votes["right-sided"] > votes["left-sided"]:
                        final_orientation = "right-sided"
                    else:
                        # Tiebreaker: use distribution method as it's most reliable for triangles
                        final_orientation = distribution_orientation
                    
                    # Draw bounding box
                    cv2.rectangle(visualization, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    
                    # Add orientation text - use white text with black outline for better visibility on any background
                    text = final_orientation.upper()
                    cv2.putText(visualization, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Thicker black outline
                    cv2.putText(visualization, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
                    
                    # Add detailed info with better visibility
                    info_text = [
                        f"Approach: {successful_approach}",
                        f"Vertices: {len(approx)}",
                        f"Left/Right ratio: {left_weight}/{right_weight}",
                        f"Centroid: ({cx},{cy})",
                        f"Geo center: ({int(geometric_center_x)},{int(y+h/2)})"
                    ]
                    
                    for i, text in enumerate(info_text):
                        y_pos = 60 + 30*i
                        cv2.putText(visualization, text, (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black outline
                        cv2.putText(visualization, text, (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # White text
                    
                    # Add triangle confidence
                    triangle_confidence = 100 if len(approx) == 3 else (75 if len(approx) == 4 else 50)
                    confidence_text = f"Triangle confidence: {triangle_confidence}%"
                    y_pos = 60 + 30*len(info_text)
                    cv2.putText(visualization, confidence_text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black outline
                    cv2.putText(visualization, confidence_text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # White text
                    
                    return final_orientation, visualization
        
        # If we got here, we couldn't identify a triangle
        # Add debug info about what we tried
        debug_text = ["No triangle detected", "Approaches tried:"]
        for i, (approach_name, _) in enumerate(approaches):
            debug_text.append(f"{i+1}. {approach_name}")
        
        # Add the debug info with good visibility
        for i, text in enumerate(debug_text):
            y_pos = 30 + 30*i
            cv2.putText(visualization, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Black outline
            cv2.putText(visualization, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # White text
        
        # Show one of the processed binary images for debugging
        debug_img = cv2.resize(color_mask, (width//3, height//3))
        visualization[10:10+debug_img.shape[0], width-10-debug_img.shape[1]:width-10] = \
            cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR) * 255
        
        return "Not a triangle", visualization

def save_analysis_json(output_path, original_path, orientation, serial):
    """Saves analysis results in JSON format."""
    data = {
        "serial": serial,
        "original_image": original_path,
        "orientation": orientation,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, "w") as json_file:
        json.dump(data, json_file)

def main():
    # Initialize classifier
    classifier = TriangleOrientationClassifier()
    
    # Create output folders
    results_folder = "classification_results"
    json_folder = "classification_json"
    create_folder(results_folder)
    create_folder(json_folder)
    
    # Get images folder
    images_folder = input("Enter folder containing triangle images (mixed left and right): ").strip()
    if not os.path.exists(images_folder):
        print(f"Error: Folder '{images_folder}' does not exist.")
        return
    
    # Process each image
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    serial = 1
    
    # Results tracking
    results = {
        "left-sided": [],
        "right-sided": [],
        "unknown": []
    }
    
    for filename in os.listdir(images_folder):
        file_path = os.path.join(images_folder, filename)
        
        # Check if it's an image file
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            print(f"\nProcessing {filename}...")
            
            # Analyze the image
            orientation, visualization = classifier.analyze_image(file_path)
            
            if visualization is not None:
                # Save the visualization image
                result_filename = f"result_{serial}_{orientation}_{filename}"
                result_path = os.path.join(results_folder, result_filename)
                cv2.imwrite(result_path, visualization)
                
                # Save the JSON data
                json_filename = f"analysis_{serial}.json"
                json_path = os.path.join(json_folder, json_filename)
                save_analysis_json(json_path, file_path, orientation, serial)
                
                print(f"Classification result: {orientation}")
                print(f"Visualization saved: {result_path}")
                
                # Keep track of results
                if orientation in results:
                    results[orientation].append(filename)
                else:
                    results["unknown"].append(filename)
                
                serial += 1
    
    # Print summary
    print("\n===== CLASSIFICATION SUMMARY =====")
    print(f"Total triangles processed: {serial-1}")
    print(f"Left-sided triangles: {len(results['left-sided'])}")
    print(f"Right-sided triangles: {len(results['right-sided'])}")
    print(f"Other results: {len(results['unknown'])}")
    
    print("\nResults saved to:")
    print(f"- Visualizations: '{results_folder}'")
    print(f"- JSON data: '{json_folder}'")
    
    # Interactive mode for individual images
    print("\n===== INTERACTIVE MODE =====")
    print("Enter image path to classify a single image (or 'q' to quit)")
    
    while True:
        image_path = input("\nImage path: ").strip()
        if image_path.lower() == 'q':
            break
            
        if os.path.exists(image_path):
            orientation, visualization = classifier.analyze_image(image_path)
            if visualization is not None:
                print(f"Classification result: {orientation}")
                
                # Display the result
                cv2.imshow("Triangle Orientation", visualization)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Error: Image not found")

if __name__ == "__main__":
    main()