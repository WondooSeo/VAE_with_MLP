import os
import cv2

aug_path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/Aug_Dataset_Thorax/"
bw_path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/BW_Dataset_Thorax/"
file_list = os.listdir(aug_path_dir)
data_num = len(file_list)
temp_stacking = []

count = 0

for img in file_list:
    dat = cv2.imread(aug_path_dir + img)
    dat_gray = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(bw_path_dir + "BW_" + img, dat_gray)
    count += 1
    print(str(count) + " / " + str(data_num) + " Stack Finished ...")