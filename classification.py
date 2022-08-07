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

def standardize(image):
    # Resize image and pre-process so that all "standard" images are the same size
    standard_im = cv2.resize(image, (600, 400))
    return standard_im

def encode(label):
    # encode day as 1, night as 0
    if(label == "day"):
        return 1
    return 0

def preprocess(image_list):

    standard_list = []
    for element in image_list:
        image = element[0]
        label = element[1]

        standardized_img = standardize(image)
        binary_label = encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_img, binary_label))

    return standard_list

#Find the average Value (V in HSV) or brightness of an image
def avg_brightness(rgb_image):
    #Convert the image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    #Add up the pixel value in V channel
    sum_brightness = np.sum(hsv_image[:,:,2])
    area = rgb_image.shape[0]*rgb_image.shape[1]

    #Calculate the average
    avg_bright = sum_brightness/area
    return avg_bright

#Read the directories
train_dir = 'Dataset/training'
test_dir = 'Dataset/testing'

# Load training data
Images = load_dataset(train_dir)
Standardized_list = preprocess(Images)

# Select a random image
img_index = random.randint(1,200)
selected_image = Standardized_list[img_index][0]
selected_label = Standardized_list[img_index][1]

#Show the image
plt.imshow(selected_image)
print("Shape:",selected_image.shape)
print("Label [1 = day, 0 = night]: " + str(selected_label))
plt.show()

# Testing average brightness levels
# Look at a number of different day and night images and think about
# what average brightness value separates the two types of images

# As an example, a "night" image is loaded in and its avg brightness is displayed
print("-"*40)
print("Check the average brightness")
for img_index in range(0,232,11):
    test_img = Standardized_list[img_index][0]
    test_label = Images[img_index][1]
    avg = avg_brightness(test_img)
    print('Average brightness:',avg,test_label)
    plt.imshow(test_img)
    plt.show()
