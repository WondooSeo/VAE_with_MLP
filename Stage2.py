import os
import csv
import numpy as np
import tensorflow as tf


path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/Dataset_VDiff/"
file_list = os.listdir(path_dir)
data_num = len(file_list)
temp_stacking = []
count = 0

for VDiff in file_list:
    dat = open(path_dir + VDiff)
    reader = csv.reader(dat)
    lines = list(reader)
    lines = np.transpose(lines)
    temp_stacking.append(lines)
    count += 1
    print(str(count) + " / " + str(data_num) + " Stack Finished ...")

V_diff = np.random.permutation(temp_stacking)
print("VDiff value shuffling Finished ...")
# print(np.shape(V_diff))


encoder_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'encoder_model_epoch300.h5'
if (os.path.exists(encoder_path)):
    encoder = tf.keras.models.load_model(encoder_path, compile=False)
    # encoder.summary()
    print("Encoder model exist & loaded ...")

else:
    print("There is no file! Check " + encoder_path + ' ...')

## Making latent vector layer code ##
loc_z_mean = len(encoder.layers) - 11
loc_z_log_var = len(encoder.layers) - 10
z_mean = encoder.layers[loc_z_mean]
z_log_var = encoder.layers[loc_z_log_var]
print(z_mean.get_weights())
z_mean_weights = z_mean.get_weights()[0]
z_mean_bias = z_mean.get_weights()[1]
print(np.shape(z_mean_weights))
print(np.shape(z_mean_bias))
z_log_var_weights = z_log_var.get_weights()[0]
z_log_var_bias = z_log_var.get_weights()[1]

z_weights = z_mean_weights + np.exp(0.5 * z_log_var_weights)
z_bias = z_mean_bias + np.exp(0.5 * z_log_var_bias)
# z_weights_init = z_weights.numpy()
# z_bias_init = z_bias.numpy()
z = tf.keras.layers.Dense(16, name="latent_z").set_weights([z_weights, z_bias])
# z = tf.keras.layers.Dense(16, kernel_initializer=z_weights_init, bias_initializer=z_bias_init, name="latent_z")
# z.trainable = False # Freeze layer
print(z)

# stage2_model = tf.keras.Sequential([tf.keras.Input(shape=(208,))])
# stage2_model.add(tf.keras.layers.Dense(256, activation="relu"))
# stage2_model.add(tf.keras.layers.Dropout(0.3))
# stage2_model.add(tf.keras.layers.Dense(128, activation="relu"))
# stage2_model.add(tf.keras.layers.Dropout(0.3))
# stage2_model.add(tf.keras.layers.Dense(64, activation="relu"))
# stage2_model.add(tf.keras.layers.Dropout(0.3))
# stage2_model.add(tf.keras.layers.Dense(32, activation="relu"))
# stage2_model.add(tf.keras.layers.Dropout(0.3))
# stage2_model.add(z)
# stage2_model.summary()

# layer1 = keras.layers.Dense(256, activation='relu')
# layer2 = keras.layers.Dense(128, activation='relu')
# layer3 = keras.layers.Dense(64, activation='relu')
# layer4 = keras.layers.Dense(32, activation='relu')
# layer5 = keras.layers.Dense(16)
# keras.backend.set_value(layer5.weights[0], z_weights)
# keras.backend.set_value(layer5.weights[1], z_bias)
# stage2_model = keras.Sequential([keras.Input(shape=208,), layer1, layer2, layer3, layer4, layer5])
# stage2_model.summary()