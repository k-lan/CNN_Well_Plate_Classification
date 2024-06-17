'''
This is a library of functions that are used for detecting, 
classifying, and analyzing the data.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os # for saving wells to files
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

'''
Basic function to display each individual well. Helpful for viewing the found circles,
its possible not all wells are properly detected, so its good for checking this. Also useful to use when labeling data for training.
'''

# Simple function to display the wells onto the screen.
# helpful for manually labeling and ensuring that the wells are
# being detected properly.
def display_wells(wells, circles_image):
    num_wells = len(wells)
    cols = 12
    rows = (num_wells // cols) + (1 if num_wells % cols else 0)
    if num_wells == 0:
        print("No wells detected.")
        return
    
    fig, axes = plt.subplots(rows, cols, figsize=(17, 17))
    for i, well in enumerate(wells):
        ax = axes[i // cols, i % cols]
        ax.imshow(cv2.cvtColor(well, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.show()


'''
Detects circles in an image and returns the wells as a list of images, the image with the detected circles, and the circles themselves.
Since there are 96 wells in a well plate, the function will try to find exactly 96 wells. This tests a wide variety of parameters until it finds 96 wells.
It is a good idea to view the output, since there is still the possibility that something random on the image is detected.
'''
def detect_circles(image_path, target_wells=96, dp=1.2, min_dist=20, param1_range=(50, 150), param2_range=(20, 50), min_radius_range=(10, 30), max_radius_range=(30, 50)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
    
    # Resize image to a standard size
    standard_size = (1280, 957)
    image = cv2.resize(image, standard_size)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Variables to store the best results
    best_circles = None
    best_wells = []
    best_image = None
    
    # sort circles row by row. first sort by y-coordinate, then by x-coordinate
    def sort_circles(circles, tolerance=20):
        circles = sorted(circles, key=lambda c: c[1])  # Sort by y-coordinate first
        circles_sorted = []
        while circles:
            row = []
            base_y = circles[0][1]
            for circle in circles[:]:
                if abs(circle[1] - base_y) < tolerance:
                    row.append(circle)
            row.sort(key=lambda c: c[0])  # Sort by x-coordinate within the row
            circles_sorted.extend(row)
            circles = [c for c in circles if abs(c[1] - base_y) >= tolerance]
        return circles_sorted
    
    # Loop over the range of parameters
    for param1 in range(param1_range[0], param1_range[1], 10):
        for param2 in range(param2_range[0], param2_range[1], 5):
            for min_radius in range(min_radius_range[0], min_radius_range[1], 5):
                for max_radius in range(max_radius_range[0], max_radius_range[1], 5):
                    # Detect edges using Canny
                    edges = cv2.Canny(blurred, param1 // 2, param1)
                    
                    # Detect circles using HoughCircles
                    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
                    
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        
                        # Sort circles row by row
                        circles = sort_circles(list(circles))
                        
                        wells = []
                        for (x, y, r) in circles:
                            if y-r >= 0 and y+r < image.shape[0] and x-r >= 0 and x+r < image.shape[1]:
                                well = image[y-r:y+r, x-r:x+r]
                                wells.append(cv2.resize(well, (50, 50)))  # Resize to a uniform size
                        
                        if len(circles) == target_wells:
                            best_circles = circles
                            best_wells = wells
                            best_image = image.copy()
                            break
            if best_circles is not None:
                break
        if best_circles is not None:
            break
    
    if best_circles is None:
        print(f"Could not find exactly {target_wells} wells. Found {len(circles)} instead.")
    
    # Display the image with detected circles, for assuring if the circles were all detected.
    # plt.figure(figsize=(10, 10))
    # plt.title('Detected Circles')
    # if best_image is not None:
    #     plt.imshow(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
    # else:
    #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    
    return best_wells, best_image, best_circles


# Create a CNN model to train. This utilized dropout layers to void overfitting
# and has 3 convolutional layers with max pooling layers in between. Filter of size 3x3
def create_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Dropout layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Dropout layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Dropout layer
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout layer
        layers.Dense(3, activation='softmax')  # Output layer with 3 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model. It will split the data into training and testing data, and then train the model.
def label_wells(image_path, model_path='best_model.keras'):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    wells, circles_image, _ = detect_circles(image_path)
    display_wells(wells, circles_image)

    wells = np.array(wells) / 255.0  # Normalize the images
    predictions = model.predict(wells)
    # print(predictions)

    class_mapping = {0: 1, 1: 0.5, 2: 0}
    predicted_classes = np.argmax(predictions, axis=1)
    # print(predicted_classes)
    predicted_values = [class_mapping[pred] for pred in predicted_classes]

    # print in csv format to be easily pasted into csv
    predictions_per_row = 12
    for i, prediction in enumerate(predicted_values):
        if (i + 1) % predictions_per_row == 0:
            print(f"{prediction}")
        else:
            print(f"{prediction},\t", end='')