import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Load data
data = pd.read_csv("D:\\NM_Data\\archive\\analog_clocks\\label.csv")

output_data_numpy = data.iloc[:, 0:2].to_numpy()

# Create dataset
import os
import tensorflow as tf

img_dir_path = "D:\\NM_Data\\archive\\analog_clocks\\images"
img_size = (300, 300)

files = next(os.walk(img_dir_path))[2]

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

output_data_numpy_hour = output_data_numpy[:, 0]
# output_data_numpy_minute = output_data_numpy[:, 1]

output_data_numpy_hour_OHE = to_categorical(output_data_numpy_hour)
# output_data_numpy_minute_OHE = to_categorical(output_data_numpy_minute)

input_train_hour, input_test_hour, output_train_hour_OHE, output_test_hour_OHE = train_test_split(files,
                                                                                                  output_data_numpy_hour_OHE,
                                                                                                  train_size=0.8,
                                                                                                  shuffle=True,
                                                                                                  random_state=14)

train_dataset_hour = tf.data.Dataset.from_tensor_slices((input_train_hour, output_train_hour_OHE))
test_dataset_hour = tf.data.Dataset.from_tensor_slices((input_test_hour, output_test_hour_OHE))

def read_image(image_path, lab):
    image = tf.io.read_file(img_dir_path + "\\" + image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, lab

train_dataset_hour = train_dataset_hour.map(read_image).batch(64)
test_dataset_hour = test_dataset_hour.map(read_image).batch(64)

# Load model
model = tf.keras.models.load_model("NM_Project_hour_OHE.h5")
model.summary()

# Predict train and test data

test_prediction_hour_OHE = model.predict(test_dataset_hour, verbose=1)
test_prediction_hour = np.argmax(test_prediction_hour_OHE, axis=1)
output_test_hour = np.argmax(output_test_hour_OHE, axis=1)
test_accuracy = np.sum(test_prediction_hour == output_test_hour) / len(output_test_hour)
print("Model accuracy: " + str(test_accuracy * 100) + "%")

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Test confusion matrix
plt.figure()
cm = confusion_matrix(output_test_hour, test_prediction_hour, normalize='true')
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(12)])
cm_display.plot()
plt.show()

