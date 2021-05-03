# import os
# from PIL import Image
# import numpy as np
# from tensorflow import keras
#
# ## Make a dataset with shuffling ##
# path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/BW_Dataset_Thorax/"
# file_list = os.listdir(path_dir)
# data_num = len(file_list)
# temp_stacking = []
# count = 0
#
# for img in file_list:
#     np_img = np.asarray(Image.open(path_dir + img))
#     temp_stacking.append(np_img)
#     count += 1
#     if count == 2:
#         break
#     print(str(count) + " / " + str(data_num) + " Stack Finished ...")
#
# print(np.shape(temp_stacking))
# temp_stacking = np.expand_dims(temp_stacking, -1)
# print(np.shape(temp_stacking))
#
# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
#
# print(np.shape(mnist_digits))

from numpy import array
a = array([[1,2,3], [2,3,4]])
print(a.shape)