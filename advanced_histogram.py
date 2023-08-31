import cv2
import numpy as np
import os

# Replace with the directory containing your images
image_directory = './gaussian_blur/novelty'
output_directory = './histogram/novelty'

# Get a list of image filenames in the directory
image_files = os.listdir(image_directory)

# Initialize an empty list to store processed images
processed_images = []

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Apply Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_image = clahe.apply(image)
        
        processed_images.append(equalized_image)
    else:
        print(f"Skipping {image_file} as it couldn't be loaded.")

# Save processed images to output directory
for i, processed_image in enumerate(processed_images):
    output_path = os.path.join(output_directory, f"processed_{i}.jpg")
    cv2.imwrite(output_path, processed_image)

print("Processed images saved to output directory.")
