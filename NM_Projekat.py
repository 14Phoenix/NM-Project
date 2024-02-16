import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Load data
data = pd.read_csv("D:\\NM_Data\\archive\\analog_clocks\\label.csv")

output_data_numpy = data.iloc[:, 0:2].to_numpy()

hour_num_of_samples = [0 for i in range(12)]
minute_num_of_samples = [0 for i in range(60)]

for i in range(len(hour_num_of_samples)):
    hour_num_of_samples[i] = (output_data_numpy[output_data_numpy[:, 0] == i]).shape[0]

for i in range(len(minute_num_of_samples)):
    minute_num_of_samples[i] = (output_data_numpy[output_data_numpy[:, 1] == i]).shape[0]

# Histogram hour
plt.figure()
plt.bar([i for i in range(12)], hour_num_of_samples, tick_label=[i for i in range(12)])
plt.show()

# Histogram minute
plt.figure(figsize=(15, 3))
plt.bar([i for i in range(60)], minute_num_of_samples, tick_label=[i for i in range(60)])
plt.show()

# print(hour_num_of_samples)
# print(minute_num_of_samples)

# print(sum(hour_num_of_samples))
# print(sum(minute_num_of_samples))

# Create dataset
import os
import tensorflow as tf

img_dir_path = "D:\\NM_Data\\archive\\analog_clocks\\images"
img_size = (300, 300)

files = next(os.walk(img_dir_path))[2]
# print(files)
# print(len(files))

dataset = tf.data.Dataset.from_tensor_slices((files, output_data_numpy.tolist()))

# for img, label in dataset:
#     print(img)
#     print(label)

def read_image(image_path, lab):
    image = tf.io.read_file(img_dir_path + "\\" + image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, lab

dataset = dataset.map(read_image).batch(32)

N = 10
plt.figure()
for img, lab in dataset.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i + 1)
        plt.imshow(img[i].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

