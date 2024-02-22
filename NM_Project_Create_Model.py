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
output_data_numpy_minute = output_data_numpy[:, 1]

output_data_numpy_hour_OHE = to_categorical(output_data_numpy_hour)
output_data_numpy_minute_OHE = to_categorical(output_data_numpy_minute)

output_data_numpy_OHE = np.concatenate((output_data_numpy_hour_OHE, output_data_numpy_minute_OHE), axis=1)

input_train, input_test, output_train_OHE, output_test_OHE = train_test_split(files,
                                                                              output_data_numpy_OHE,
                                                                              train_size=0.8,
                                                                              shuffle=True,
                                                                              random_state=14)

def generator_train_data():
    num_of_files = len(input_train)
    i = 0
    while i < num_of_files:
        image = tf.io.read_file(img_dir_path + "\\" + input_train[i])
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.expand_dims(image, axis=0)
        yield image, (tf.expand_dims(output_train_OHE[i, :12], axis=0), tf.expand_dims(output_train_OHE[i, 12:], axis=0))
        i += 1

train_dataset = tf.data.Dataset.from_generator(generator_train_data, output_signature=(
    tf.TensorSpec(shape=(None, 300, 300, 3), dtype=tf.float32),
    (tf.TensorSpec(shape=(None, 12), dtype=tf.float32), tf.TensorSpec(shape=(None, 60), dtype=tf.float32))
))

def generator_test_data():
    num_of_files = len(input_test)
    i = 0
    while i < num_of_files:
        image = tf.io.read_file(img_dir_path + "\\" + input_test[i])
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.expand_dims(image, axis=0)
        yield image, (tf.expand_dims(output_test_OHE[i, :12], axis=0), tf.expand_dims(output_test_OHE[i, 12:], axis=0))
        i += 1

test_dataset = tf.data.Dataset.from_generator(generator_test_data, output_signature=(
    tf.TensorSpec(shape=(None, 300, 300, 3), dtype=tf.float32),
    (tf.TensorSpec(shape=(None, 12), dtype=tf.float32), tf.TensorSpec(shape=(None, 60), dtype=tf.float32))
))

# Create a model
from keras import Model
from keras import layers
from keras.regularizers import l2

# Hour, minute model input layer
model_input = layers.Input(shape=(300, 300, 3), name='model_input', dtype=tf.float32)

# Hour sub-model
layer = layers.RandomContrast(0.1, name='hour_sub_model_input')(model_input)
layer = layers.RandomBrightness(0.2)(layer)
layer = layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(layer)
layer = layers.MaxPooling2D()(layer)
layer = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(layer)
layer = layers.MaxPooling2D()(layer)
layer = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(layer)
layer = layers.MaxPooling2D()(layer)
layer = layers.Flatten()(layer)

layer = layers.Dropout(0.3)(layer)
layer = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(layer)
hour_sub_model_output = layers.Dense(12, activation='softmax', name='hour_sub_model_output')(layer)

# Minute sub-model
layer = layers.RandomContrast(0.1, name='minute_sub_model_input')(model_input)
layer = layers.RandomBrightness(0.2)(layer)
layer = layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(layer)
layer = layers.MaxPooling2D()(layer)
layer = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(layer)
layer = layers.MaxPooling2D()(layer)
layer = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(layer)
layer = layers.MaxPooling2D()(layer)
layer = layers.Flatten()(layer)

layer = layers.Dropout(0.3)(layer)
layer = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(layer)
minute_sub_model_output = layers.Dense(60, activation='softmax', name='minute_sub_model_output')(layer)

# Create hour, minute model
model = Model(model_input,(hour_sub_model_output, minute_sub_model_output))

model.summary()

# Compile and train the model
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping

model.compile(optimizer='adam',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

stop_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_dataset,
                    epochs=75,
                    batch_size=64,
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

model.save('NM_Project_super_model_OHE.h5')