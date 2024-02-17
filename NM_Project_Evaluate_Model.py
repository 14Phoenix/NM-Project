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

input_train, input_test, output_train, output_test = train_test_split(files,
                                                                      output_data_numpy,
                                                                      train_size=0.8,
                                                                      shuffle=True,
                                                                      random_state=14)

train_dataset = tf.data.Dataset.from_tensor_slices((input_train, output_train))
test_dataset = tf.data.Dataset.from_tensor_slices((input_test, output_test))

def read_image(image_path, lab):
    image = tf.io.read_file(img_dir_path + "\\" + image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, lab

train_dataset = train_dataset.map(read_image).batch(64)
test_dataset = test_dataset.map(read_image).batch(64)

# Load model
model = tf.keras.models.load_model("NM_Project.h5")
model.summary()

print("Accuracy: " + str(100 * model.evaluate(test_dataset, verbose=1)[1]) + "%")