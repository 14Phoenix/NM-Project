import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# Show one of each hour class
img_dir_path = "D:\\NM_Data\\archive\\analog_clocks\\images"

plt.figure()
fig, ax = plt.subplots(2, 6)

hour_class_examples = ["00059.jpg", "00654.jpg", "01369.jpg", "00800.jpg", "00398.jpg", "01496.jpg",
                       "00824.jpg", "00233.jpg", "01131.jpg", "01315.jpg", "00006.jpg", "00196.jpg"]
for i in range(12):
    img = mpimg.imread(img_dir_path + "\\" + hour_class_examples[i])
    ax[i//6, i%6].axis("off")
    ax[i//6, i%6].imshow(img)
    ax[i//6, i%6].set_title(str(i))
fig.show()

# Show one of each minute class
plt.figure()
fig, ax = plt.subplots(10, 6)

for i in range(60):
    indices = np.where(output_data_numpy[:, 1] == i)
    index = str(indices[0][0])
    img = mpimg.imread(img_dir_path + "\\" + ("0" * (5 - len(index))) + index + ".jpg")
    ax[i//6, i%6].axis("off")
    ax[i//6, i%6].imshow(img)
fig.show()