import os
from PIL import Image
import numpy as np

## Make a dataset with shuffling ##
path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/BW_Dataset_Thorax/"
file_list = os.listdir(path_dir)
data_num = len(file_list)
temp_stacking = []
count = 0

for img in file_list:
    np_img = np.asarray(Image.open(path_dir + img))
    # tensor_img = tf.convert_to_tensor(np_img)
    # temp_stacking.append(tensor_img)
    temp_stacking.append(np_img)
    count += 1
    print(str(count) + " / " + str(data_num) + " Stack Finished ...")