import cv2
import os
#shreya humari new teacher!!
# Define the directory where your input images are located
input_directory = './gaussian_blur/leaf spot'

# Define the directory where the equalized images will be saved
output_directory = './histogram/leaf spot'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to apply histogram equalization to an image and save it
def apply_histogram_equalization(image_path, output_path):
    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)
    
    # Save the equalized image
    cv2.imwrite(output_path, equalized_image)

# Loop through the images in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        # Create full paths for input and output images
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, filename)
        
        # Apply histogram equalization and save the equalized image
        apply_histogram_equalization(input_image_path, output_image_path)
