# This file is to calculate all the training data with index = 1, 2, 3, 4, (!=5) average rating mu, respectively.
# Created by Yu Wu at 05/05/2019.
import pandas as pd
import numpy as np

df_train_no5_with_date = pd.read_table("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_no5_with_date.dta", delim_whitespace=True, header=None, names=['user', 'movie', 'date', 'rating'])
df_train_no5_with_date_arr = df_train_no5_with_date.values
mu_no5 = np.mean(df_train_no5_with_date_arr[:, 3]) # ratings
print("mu of training data no index == 5: ", mu_no5)

df_train_no_probe_with_date = pd.read_table("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_no_probe_with_date.txt", delim_whitespace=True, header=None, names=['user', 'movie', 'date', 'rating'])
df_train_no_probe_with_date_arr = df_train_no_probe_with_date.values
mu_no_probe = np.mean(df_train_no_probe_with_date_arr[:, 3]) # ratings
print("mu of training data no probe (index != 4 && index != 5): ", mu_no_probe)

for i in range(1, 5):
	df_train_with_date = pd.read_table("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_" + str(i) + "_with_date.dta", delim_whitespace=True, header=None, names=['user', 'movie', 'date', 'rating'])
	df_train_with_date_arr = df_train_with_date.values # numpy array
	mu = np.mean(df_train_with_date_arr[:, 3]) # ratings
	print("mu of training data " + str(i) + " : ", mu)