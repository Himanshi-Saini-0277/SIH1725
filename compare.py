import cv2
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd
from tabulate import tabulate

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def extract_features(image_path):
    image = preprocess_image(image_path)
    features = model.predict(image)
    return features

def calculate_image_difference(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return diff, score

def generate_insights(diff_score):
    if diff_score > 0.8:
        return ["Minimal construction progress detected.",
                "The site shows only slight modifications."]
    elif 0.5 < diff_score <= 0.8:
        return ["Moderate construction progress detected.",
                "Significant changes observed in key areas."]
    else:
        return ["Substantial construction progress detected.",
                "Major structural differences found between the images.",
                "Considerable advancement in construction."]

def generate_comparison_report(previous_image_path, current_image_path):
    
    # Check if paths are correct and files exist
    if not os.path.exists(previous_image_path):
        raise FileNotFoundError(f"Previous image not found at {previous_image_path}")
    if not os.path.exists(current_image_path):
        raise FileNotFoundError(f"Current image not found at {current_image_path}")
    
    # Load images
    previous_image = cv2.imread(previous_image_path)
    current_image = cv2.imread(current_image_path)
    
    # Verify if the images are loaded successfully
    if previous_image is None:
        raise FileNotFoundError(f"Failed to load previous image from {previous_image_path}")
    if current_image is None:
        raise FileNotFoundError(f"Failed to load current image from {current_image_path}")
    
    # Calculate image difference and similarity score
    diff_image, diff_score = calculate_image_difference(previous_image, current_image)
    
    # Generate textual insights based on the similarity score
    insights = generate_insights(diff_score)
    
    # Save the difference image
    diff_image_path = r'Images/difference_image.jpeg'
    cv2.imwrite(diff_image_path, diff_image)

    # Prepare the data for the terminal output
    data = [
        ["Previous Image", previous_image_path],
        ["Current Image", current_image_path],
        ["Difference Image", diff_image_path],
        ["AI Insights", "\n".join(insights)]
    ]
    
    # Display the data in tabular form in the terminal
    print(tabulate(data, headers=["Description", "Details"], tablefmt="grid"))

# Example usage
previous_image_path = r'Images\previous_site_image.jpeg'
current_image_path = r'Images\current_site_image.jpeg'

print(f"Previous image exists: {os.path.exists(previous_image_path)}")
print(f"Current image exists: {os.path.exists(current_image_path)}")

comparison_report = generate_comparison_report(previous_image_path, current_image_path)
