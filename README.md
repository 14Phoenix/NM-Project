# Analog clock CNN
This project was created as part of a school assignment for 13S053NM at School for Electrical Engineering of the University of Belgrade. The goal was to create a CNN (Convolutional Neural Network) using a dataset of our choice. The dataset chosen for this project can be found using this [link](https://www.kaggle.com/datasets/shivajbd/analog-clocks/data). This CNN can recognize the number of hours and minutes from a given image of an analog clock.

## Architecture
The CNN consists of two smaller CNNs, one for recognizing the number of hours and one for recognizing the number of minutes from a given image.

The model for recognizing the number of hours is comprised of the following layers:

| Layer number |       Layer       |
|:------------:|:-----------------:|
|      1       |  Random Contrast  |
|      2       | Random Brightness |
|      3       |   Convolutional   |
|      4       |    Max Pooling    |
|      5       |   Convolutional   |
|      6       |    Max Pooling    |
|      7       |   Convolutional   |
|      8       |    Max Pooling    |
|      9       |      Flatten      |
|      10      |      Dropout      |
|      11      |       Dense       |
|      12      |       Dense       |

The model for recognizing the number of minutes is comprised of the following layers:

| Layer number |       Layer       |
|:------------:|:-----------------:|
|      1       |  Random Contrast  |
|      2       | Random Brightness |
|      3       |   Convolutional   |
|      4       |    Max Pooling    |
|      5       |   Convolutional   |
|      6       |    Max Pooling    |
|      7       |   Convolutional   |
|      8       |    Max Pooling    |
|      9       |      Flatten      |
|      10      |      Dropout      |
|      11      |       Dense       |
|      12      |       Dense       |
|      13      |       Dense       |

## Implementation
Both models are implemented using TensorFlow, Keras and Scikit-learn. Scripts NM_Project_Create_Hour_Model.py and NM_Project_Create_Minute_Model.py create models for recognizing the number of hours and the number of minutes respectively.

## Training
For easier image-label matching image filenames have been renamed so they have leading zeroes.

During training Random Contrast and Random Brightness layers are used to augment the image. To prevent overfitting a Dropout layer, L2 regularization and early stopping is used for both models.