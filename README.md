# Day vs Night image classifier
This is an image classifier without machine learning or deep learning algorithms. The classifier achieves impressive results, with 92.5% accuracy and 92.68% F1-score.
## Reference
This project is guided from a hands-on tutorial named "[Day-Night Image Classifier](https://github.com/arunnthevapalan/day-night-classifier)" by Arunn Thevapalan.
## Pre-requisites
This project was developed using Python with following packages:
- numpy
- pandas
- matplotlib
- opencv-python
- scikit-learn

These packages can be installed with the following pip command:  

 ``` pip install -r requirements.txt ```

## Dataset description
The dataset contains 400 RGB color images from [AMOS](https://mvrl.cse.wustl.edu/datasets/amos/) dataset (Archive of Many Outdoor Scenes). The numbers of each category are equal: 200 day images and 200 night images. This gives us a balanced dataset.  
Each category is divided into 2 sets: the training set with 120 instances and the test set with 80 instances.
## Approach
Step 1: Load and visualize the data  
- Visualization helps us notice a feature: the night images are generally darker than the day images.  

Step 2: Preprocess the data
- Resize all the images to a fixed size and encode the labels.  

Step 3: Feature extracction
- Convert the images from RGB to HSV color space. Add up the pixels in the V channel (a measure of brightness), then divide that sum by the image area to get the average brightness value of an image.  

Step 4: Build the classifier
- Check the average brightness values of various images  the training dataset to set a threshold, which separates the two classes  

Step 5: Evaluate the model and optimize
- Calculate the accuracy and F1-score of the classifier on the test dataset, using scikit-learn built-in functions: ``` accuracy_score ``` and ``` f1_score ```
