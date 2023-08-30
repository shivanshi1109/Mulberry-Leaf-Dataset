import cv2
import numpy as np
import os

# Replace with the directory containing your images
image_directory = 'C:/Users/shiva/Downloads/Mulberry Leaf Dataset/Mulberry Leaf Dataset/Training set/training dataset'

# Get a list of image filenames in the directory
image_files = os.listdir(image_directory)

# Initialize an empty list to store grayscale images
grayscale_images = []

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    # Load the image in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray_image is not None:
        grayscale_images.append(gray_image)
    else:
        print(f"Skipping {image_file} as it couldn't be loaded.")

# Convert the list of images to a NumPy array
batch_grayscale_images = np.array(grayscale_images)

# Print the shape of the batch grayscale images array
print("Shape of Batch Grayscale Images:", batch_grayscale_images.shape)
