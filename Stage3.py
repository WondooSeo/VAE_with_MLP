import os
import csv
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

stage2_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'stage2_MLP_EIT_FER.h5'
if (os.path.exists(stage2_path)):
    stage2 = tf.keras.models.load_model(stage2_path, compile=False)
    # encoder.summary()
    print("MLP model exist & loaded ...")

else:
    print("There is no file! Check " + stage2_path + ' ...')

decoder_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'decoder_EIT_FER.h5'
if (os.path.exists(decoder_path)):
    decoder = tf.keras.models.load_model(decoder_path, compile=False)
    # encoder.summary()
    print("Encoder model exist & loaded ...")

else:
    print("There is no file! Check " + decoder_path + ' ...')

test_VDiff_dir = "C:/Users/mirac/Documents/Pycharm/VAE/EIT_V/EIT_V_170.csv"
# test_VDiff_dir = "C:/Users/mirac/Documents/Pycharm/VAE/EIT_V/EIT_V_460.csv"
dat = open(test_VDiff_dir)
reader = csv.reader(dat)
lines = list(reader)
lines = np.squeeze(lines)
test_VDiff = [float(i) for i in lines]
test_VDiff = np.reshape(np.asarray(test_VDiff), (1, 208))


st2 = stage2.predict(test_VDiff)
result = decoder.predict(st2) * 255
# print(result)
temp = np.reshape(result, (128, 128))
result = np.reshape(result, (128, 128, 1))
plt.imshow(result)
plt.axis('off')
plt.show()