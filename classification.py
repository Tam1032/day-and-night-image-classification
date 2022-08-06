import os
import glob #for loading images from a directory
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np

def load_dataset(dir):
    '''This function loads in images and their labels and places them in a list
    imgs_list[0][:] will be the first image-label pair in the list'''

    imgs_list = []
    labels = ["day", "night"]

    # Iterate through each color folder
    for label in labels:

        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "dir/label/*"
        for file in glob.glob(os.path.join(dir, label, "*")):

            # Read in the image
            img = mpimg.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if img is not None:
                # Append the image, and it's type (day,night) to the image list
                imgs_list.append((img, label))

    return imgs_list

train_dir = 'Dataset/training'
test_dir = 'Dataset/testing'

# Load training data
images = load_dataset(train_dir)
# Select a random image
img_index = random.randint(1,200)
selected_image = images[img_index][0]
#Show the image
plt.imshow(selected_image)
plt.show()
