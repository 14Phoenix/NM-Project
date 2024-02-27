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
output_data_numpy_hour_OHE = to_categorical(output_data_numpy_hour)

input_train_hour, input_test_hour, output_train_hour_OHE, output_test_hour_OHE = train_test_split(files,
                                                                                                  output_data_numpy_hour_OHE,
                                                                                                  train_size=0.8,
                                                                                                  shuffle=True,
                                                                                                  random_state=14)

train_dataset = tf.data.Dataset.from_tensor_slices((input_train_hour, output_train_hour_OHE))
test_dataset = tf.data.Dataset.from_tensor_slices((input_test_hour, output_test_hour_OHE))

def read_image(image_path, lab):
    image = tf.io.read_file(img_dir_path + "\\" + image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, lab

train_dataset = train_dataset.map(read_image).batch(64)
test_dataset = test_dataset.map(read_image).batch(64)

# Create a model
from keras import Sequential
from keras import layers
from keras.regularizers import l2

model = Sequential([
    layers.RandomContrast(0.1, input_shape=(300, 300, 3)),
    layers.RandomBrightness(0.2),
    layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dense(12, activation='softmax')
])

model.summary()

# Compile and train the model
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping

model.compile(optimizer='adam',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

stop_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_dataset,
                    epochs=150,
                    validation_data=test_dataset,
                    callbacks=[stop_early],
                    verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print('Train accuracy: ', acc)
print('Validation accuracy: ', val_acc)

print('Train loss: ', loss)
print('Validation loss: ', val_loss)

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

model.save('NM_Project_Hour_OHE.h5')