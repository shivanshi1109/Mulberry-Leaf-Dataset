import os
import csv
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

data_dir = "./5.data_formatting"
disease_labels = ["disease free leaves", "leaf rust", "leaf spot"]
num_classes = len(disease_labels)

images = []
labels = []

for label_idx, label in enumerate(disease_labels):
    label_dir = os.path.join(data_dir, label)
    for image_filename in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_filename)
        image = load_img(image_path, target_size=(224, 224))  # Resize to desired size
        image_array = img_to_array(image)
        images.append(image_array)
        labels.append(label_idx)
#output directory
# output_directory = '.label/encoding'
X = np.array(images)
y = to_categorical(labels, num_classes)
import pandas as pd

# Sample encoded labels and image data
encoded_labels = [y]
images_data = [X]
# Create a DataFrame
data = {'Image Filename': images_data, 'Encoded Label': encoded_labels}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('encoded_labels.csv', index=False)

print("CSV file 'encoded_labels.csv' has been created.")




