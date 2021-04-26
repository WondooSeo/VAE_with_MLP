import os
import csv
import numpy as np



path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/Dataset_VDiff/"
file_list = os.listdir(path_dir)
data_num = len(file_list)
temp_stacking = []

count = 0

print(file_list[0])

for VDiff in file_list:
    dat = open(path_dir + VDiff)
    reader = csv.reader(dat)
    lines = list(reader)
    temp_stacking.append(lines)
    count += 1
    print(str(count) + " / " + str(data_num) + " Stack Finished ...")

print(np.shape(temp_stacking))