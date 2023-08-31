import cv2
import numpy as np
import os

# Replace with the directory containing your images
image_directory = './histogram/novelty'
output_directory = './minmax/novelty'

# Get a list of image filenames in the directory
image_files = os.listdir(image_directory)

# Set the target size for resizing
target_size = (224, 224)

# Initialize an empty list to store processed images
processed_images = []

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Resize the grayscale image
        resized_image = cv2.resize(image, target_size)
        
        # Print statistics before scaling
        print("Before scaling:")
        print("Min pixel value:", np.min(resized_image))
        print("Max pixel value:", np.max(resized_image))
        
        # Min-Max scaling to [0, 1]
        normalized_image = (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image))
        
        # Print statistics after scaling
        print("After scaling:")
        print("Min pixel value:", np.min(normalized_image))
        print("Max pixel value:", np.max(normalized_image))
        
        processed_images.append(normalized_image)
    else:
        print(f"Skipping {image_file} as it couldn't be loaded.")

# Save processed images to output directory
for i, processed_image in enumerate(processed_images):
    output_path = os.path.join(output_directory, f"processed_{i}.jpg")
    
    # Convert pixel values back to [0, 255] range for saving as image
    scaled_image = (processed_image * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, scaled_image)

print("Processed images saved to output directory.")
