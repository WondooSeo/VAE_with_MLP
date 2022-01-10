import os
import csv
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


bw_dataset_thorax_path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/EIT_FER/"
vdiff_path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/EIT_V/"
bw_dataset_thorax_file_list = os.listdir(bw_dataset_thorax_path_dir)
vdiff_file_list = os.listdir(vdiff_path_dir)
bw_dataset_thorax_data_num = len(vdiff_file_list)
vdiff_data_num = len(vdiff_file_list)
bw_dataset_thorax_stacking = []
vdiff_stacking = []
bw_count = 0
vdiff_count = 0


if bw_dataset_thorax_data_num == vdiff_data_num:
    data_num = str(vdiff_data_num)

    for VDiff in vdiff_file_list:
        dat = open(vdiff_path_dir + VDiff)
        reader = csv.reader(dat)
        lines = list(reader)
        lines = np.squeeze(lines)
        float_lines = [float(i) for i in lines]
        float_lines = np.transpose(np.reshape(np.asarray(float_lines), -1))
        vdiff_stacking.append(float_lines)
        vdiff_count += 1
        print(str(vdiff_count) + " / " + data_num + " VDiff Stack Finished ...")

    for BWimg in bw_dataset_thorax_file_list:
        np_img = np.asarray(Image.open(bw_dataset_thorax_path_dir + BWimg))
        bw_dataset_thorax_stacking.append(np_img)
        bw_count += 1
        print(str(bw_count) + " / " + str(data_num) + " Thorax Stack Finished ...")

else:
    print("Data numbers are not equal! Try again ...")
    exit(3)


# Normalize pixel value & expand dims to fit the input of encoder
bw_dataset_thorax_stacking = np.expand_dims(bw_dataset_thorax_stacking, -1)

encoder_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'encoder_EIT_FER.h5'
if (os.path.exists(encoder_path)):
    encoder = tf.keras.models.load_model(encoder_path, compile=False)
    # encoder.summary()
    print("Encoder model exist & loaded ...")

else:
    print("There is no file! Check " + encoder_path + ' ...')


encoded = encoder.predict(bw_dataset_thorax_stacking)
# print(np.shape(encoded)) → Result : (3, 576, 16)
# Latent vector z is on third (∵ [z_mean, z_log_var, z])
encoder_result = np.asarray(encoded[2])


# Extremely important!!!
# Must shpae (num, 1, dim), not (num, dim, 1)
encoder_result = np.expand_dims(encoder_result, 1)
vdiff_stacking = np.expand_dims(vdiff_stacking, 1)


x_train, x_test, y_train, y_test = train_test_split(vdiff_stacking, encoder_result, shuffle=True, test_size=0.3)
print("Data split Finished ...")


# # NO USE
# # ## Making latent vector layer code ##
# # loc_z_mean = len(encoder.layers) - 11
# # loc_z_log_var = len(encoder.layers) - 10
# # z_mean = encoder.layers[loc_z_mean]
# # z_log_var = encoder.layers[loc_z_log_var]
# # z_mean_weights = z_mean.get_weights()[0]
# # z_mean_bias = z_mean.get_weights()[1]
# # z_log_var_weights = z_log_var.get_weights()[0]
# # z_log_var_bias = z_log_var.get_weights()[1]
# # z_weights = z_mean_weights + np.exp(0.5 * z_log_var_weights)
# # z_bias = z_mean_bias + np.exp(0.5 * z_log_var_bias)


def create_model():
    MLP_model = tf.keras.Sequential(name='MLP_Model')
    MLP_model.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(208,)))
    MLP_model.add(tf.keras.layers.Dropout(0.5))
    MLP_model.add(tf.keras.layers.Dense(128, activation="relu"))
    MLP_model.add(tf.keras.layers.Dropout(0.5))
    MLP_model.add(tf.keras.layers.Dense(64, activation="relu"))
    MLP_model.add(tf.keras.layers.Dropout(0.5))
    MLP_model.add(tf.keras.layers.Dense(32, activation="relu"))
    MLP_model.add(tf.keras.layers.Dropout(0.5))
    MLP_model.add(tf.keras.layers.Dense(16, activation="relu"))
    MLP_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse', 'accuracy'])
    MLP_model.summary()
    return MLP_model


stage2_model = create_model()
stage2_model.fit(x_train, y_train, validation_split=0.3, epochs=300, batch_size=10, verbose=1, shuffle=True)
test_scores = stage2_model.evaluate(x_test, y_test, verbose=0)
print("Test Loss : ", test_scores[0])
print("Test Accuracy : ", test_scores[1])

stage2_model_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + "stage2_MLP_EIT_FER.h5"
stage2_model.save(stage2_model_path)