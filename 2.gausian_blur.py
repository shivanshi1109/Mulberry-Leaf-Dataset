import cv2
import os

# Define the input directory where your noisy images are located
input_directory = './resize_normalise_grayscale/leaf spot'

# Define the output directory where the blurred images will be saved
output_directory = './gaussian_blur/leaf spot'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to apply Gaussian blur to an image
def apply_gaussian_blur(image_path, output_path, kernel_size=(5, 5), sigmaX=0):
    # Read the image
    image = cv2.imread(image_path)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX)
    
    # Save the blurred image to the output directory
    cv2.imwrite(output_path, blurred_image)

# Loop through the images in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        # Create full paths for input and output images
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, filename)
        
        # Apply Gaussian blur to the image
        apply_gaussian_blur(input_image_path, output_image_path)
