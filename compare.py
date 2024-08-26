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
        return ["1. Minimal construction progress detected.",
                "2. The site shows only slight modifications."]
    elif 0.5 < diff_score <= 0.8:
        return ["1. Moderate construction progress detected.",
                "2. Significant changes observed in key areas."]
    else:
        return ["1. Substantial construction progress detected.",
                "2. Major structural differences found between the images.",
                "3. Considerable advancement in construction."]

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

    new_html_content = f"""

    <!DOCTYPE html>
        <html lang="en">

        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="css/style.css">
            <title>Home ImageHub</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    color: #e307f8;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    align-items: center;
                    overflow: hidden;
                    background: url(progress-bg.jpg);
                    backdrop-filter: blur(2px);
                    width: 100%;
                    background-size: cover;
                }}

                header {{
                    width: 100%;
                    height: 70px;
                    color: #ccc;
                    padding: 30px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    position: relative;
                }}

                .logo {{
                    display: flex;
                    align-items: center;
                    padding: 5px;
                }}

                .logo img {{
                    width: 150px;
                    height: auto;
                    margin-right: 20px;
                    margin-left: -18px;
                }}

                .space {{
                    width: 170px;
                    height: 130px;
                }}

                .nav-links {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    position: relative;
                    height: 100%;
                    width: 100%;
                    color: #fff;
                    text-decoration: none;
                    font-size: 45px;
                    transition: color 0.3s;
                    margin-left: -50px;
                }}

                .main-content {{
                    width: 100%;
                    height: 80vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}

                .live-display {{
                    margin-top: -20px;
                    width: 80%;
                    background-color: rgba(224, 224, 224, 0.308);
                    border-radius: 10px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    color: black;
                    overflow: auto; /* Adds scrollbars if content overflows */
                    padding: 20px; /* Adds padding inside the live-display */
                }}
            </style>
        </head>

        <body>
            <header>
                <div class="logo">
                    <img src="logo.png" alt="Site Logo">
                </div>
                <div class="nav-links">Progress Tracker</div>
                <div class="space"></div>
            </header>

            <div class="main-content">
                <div class="img" style="height: 100%; background-size: cover; width: 100%; display: flex; flex-direction: column; align-items: center;">
                    <div class="live-display">
                        <table>
                            <br><br><br><br><br><br><br><br><br><br>
                            <tr>
                                <td><strong>AI Insights:</strong></td>
                                <td>{'<br>'.join(insights)}</td>
                            </tr>
                            <tr>
                                <td><strong style="margin-top: 250px;">Previous Image:</strong></td>
                                <td><img src="Images/previous_site_image.jpeg" style="width: 200px; height: 200px; margin-left: 200px; margin-top: 50px;"></td>
                            </tr>
                            <tr>
                                <td><strong>Current Image:</strong></td>
                                <td><img src="Images/current_site_image.jpeg" style="width: 200px; height: 200px; margin-left: 200px;"></td>
                            </tr>
                            <tr>
                                <td><strong>Differences Image:</strong></td>
                                <td><img src="Images/difference_image.jpeg" style="width: 200px; height: 200px; margin-left: 200px;"></td>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>
        </body>

        </html>
    """
    progress_file_path = "progress.html"
    with open(progress_file_path, "w") as file:
        file.write(new_html_content)

    print("Report generated and written to progress.html")
    
    # Display the data in tabular form in the terminal
    print(tabulate(data, headers=["Description", "Details"], tablefmt="grid"))

# Example usage
previous_image_path = r'Images\previous_site_image.jpeg'
current_image_path = r'Images\current_site_image.jpeg'

print(f"Previous image exists: {os.path.exists(previous_image_path)}")
print(f"Current image exists: {os.path.exists(current_image_path)}")

comparison_report = generate_comparison_report(previous_image_path, current_image_path)
