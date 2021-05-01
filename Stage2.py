import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


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


x_train, x_test = train_test_split(V_diff, random_state=999, test_size=0.3)
x_test, x_val = train_test_split(V_diff, random_state=999, test_size=0.2)
print("VDiff split Finished ...")


encoder_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'encoder_model_epoch300.h5'
if (os.path.exists(encoder_path)):
    encoder = tf.keras.models.load_model(encoder_path, compile=False)
    # encoder.summary()
    print("Encoder model exist & loaded ...")

else:
    print("There is no file! Check " + encoder_path + ' ...')

# NO USE
# ## Making latent vector layer code ##
# loc_z_mean = len(encoder.layers) - 11
# loc_z_log_var = len(encoder.layers) - 10
# z_mean = encoder.layers[loc_z_mean]
# z_log_var = encoder.layers[loc_z_log_var]
# z_mean_weights = z_mean.get_weights()[0]
# z_mean_bias = z_mean.get_weights()[1]
# z_log_var_weights = z_log_var.get_weights()[0]
# z_log_var_bias = z_log_var.get_weights()[1]
#
# z_weights = z_mean_weights + np.exp(0.5 * z_log_var_weights)
# z_bias = z_mean_bias + np.exp(0.5 * z_log_var_bias)


def create_model():
    MLP_model = tf.keras.Sequential()
    MLP_model.add(tf.keras.layers.Input(shape=(208, 1)))
    MLP_model.add(tf.keras.layers.Dense(256, activation="relu"))
    MLP_model.add(tf.keras.layers.Dropout(0.3))
    MLP_model.add(tf.keras.layers.Dense(128, activation="relu"))
    MLP_model.add(tf.keras.layers.Dropout(0.3))
    MLP_model.add(tf.keras.layers.Dense(64, activation="relu"))
    MLP_model.add(tf.keras.layers.Dropout(0.3))
    MLP_model.add(tf.keras.layers.Dense(32, activation="relu"))
    MLP_model.add(tf.keras.layers.Dropout(0.3))
    MLP_model.add(tf.keras.layers.Dense(16, activation="relu"))
    MLP_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse', 'accuracy'])
    MLP_model.summary()
    return MLP_model


stage2_model = create_model()
stage2_model.fit(x_train, epochs=1, batch_size=18, validation_data=(x_val,))
print(stage2_model.predict(x_test))