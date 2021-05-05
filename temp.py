import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

decoder_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'decoder_model_BW_epoch300.h5'
if (os.path.exists(decoder_path)):
    decoder = tf.keras.models.load_model(decoder_path, compile=False)
    # encoder.summary()
    print("Encoder model exist & loaded ...")

else:
    print("There is no file! Check " + decoder_path + ' ...')

st2 = z_sample = np.array([[10, -12, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1]])
result = decoder.predict(st2)
print(result)
plt.imshow(np.reshape(result, (128, 128, 1)))
plt.axis('off')
plt.show()