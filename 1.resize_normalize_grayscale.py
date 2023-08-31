import cv2
import numpy as np
import os

# Replace with the directory containing your images
image_directory = './Mulberry Data/Leaf Spot'
output_directory = './resize_normalise_grayscale/leaf spot'

# Get a list of image filenames in the directory
image_files = os.listdir(image_directory)

# Set the target size for resizing
target_size = (224, 224)

# Initialize an empty list to store processed grayscale images
processed_images = []

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Resize the grayscale image
        resized_image = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 255] (for saving as an image)
        normalized_image = (resized_image * 255).astype(np.uint8)
        
        processed_images.append(normalized_image)
    else:
        print(f"Skipping {image_file} as it couldn't be loaded.")

# Save processed images to output directory
for i, processed_image in enumerate(processed_images):
    output_path = os.path.join(output_directory, f"processed_{i}.jpg")
    cv2.imwrite(output_path, processed_image)

print("Processed images saved to output directory.")
