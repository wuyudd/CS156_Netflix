# This file is to read in all the data and separate them according to the index.
# Created by Yu Wu at 05/05/2019.

import pandas as pd
import numpy as np

# read in all the data and corresponding index
print("++++++++++++++++++ Start Loading all.dta ... ++++++++++++++++++")
df_all_with_date = pd.read_table('/Volumes/Disk1/ywwu/courses/cs156b/data/um/all.dta', delim_whitespace=True, header=None, names=['user', 'movie', 'date', 'rating'])
df_all_no_date = df_all_with_date.drop(columns=['date'], inplace=False)
df_all_with_date_arr = df_all_with_date.values
df_all_no_date_arr = df_all_no_date.values
print("++++++++++++++++++ Now all.dta loaded successfully! ++++++++++++++++++")

# read in qual data
print("++++++++++++++++++ Start Loading qual.dta ... ++++++++++++++++++")
df_qual_with_date = pd.read_table('/Volumes/Disk1/ywwu/courses/cs156b/data/um/qual.dta', delim_whitespace=True, header=None, names=['user', 'movie', 'date'])
df_qual_no_date = df_qual_with_date.drop(columns=['date'], inplace=False)
df_qual_with_date_arr = df_qual_with_date.values
df_qual_no_date_arr = df_qual_no_date.values
print("++++++++++++++++++ Now qual.dta loaded successfully! ++++++++++++++++++")

# save qual data
np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/qual_with_date.txt", df_qual_with_date_arr, delimiter="\t", fmt='%10d') 
np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/qual_no_date.txt", df_qual_no_date_arr, delimiter="\t", fmt='%10d')


# read in the corresponding index
print("++++++++++++++++++ Start Loading all.idx ... ++++++++++++++++++")
df_idx = pd.read_table('/Volumes/Disk1/ywwu/courses/cs156b/data/um/all.idx', delim_whitespace=True, header=None, names = ['set'])
print("++++++++++++++++++ Now all.idx loaded successfully! ++++++++++++++++++")

print("++++++++++++++++++ Separate all the training data with index != 5 ... ++++++++++++++++++")
train_no5_idx = df_idx[df_idx.set != 5].index.values # index = 1, 2, 3, 4
#train_no5_idx = df_idx[df_idx.set <= 3].index.values # index = 1, 2, 3
df_train_no5_with_date = df_all_with_date.loc[train_no5_idx, :] # user, movie, date, rating
df_train_no5_with_date_arr = df_train_no5_with_date.values

df_train_no5_no_date = df_all_no_date.loc[train_no5_idx, :]
df_train_no5_no_date_arr = df_train_no5_no_date.values

#np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_no5_with_date.dta", df_train_no5_with_date_arr, delimiter="\t", fmt='%10d')
np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_no5_with_date.txt", df_train_no5_with_date_arr, delimiter="\t", fmt='%10d')
#np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_no_probe_with_date.txt", df_train_no5_with_date_arr, delimiter="\t", fmt='%10d')

np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_no5_no_date.dta", df_train_no5_no_date_arr, delimiter="\t", fmt='%10d')
#np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_no_probe_no_date.txt", df_train_no5_no_date_arr, delimiter="\t", fmt='%10d')
print("++++++++++++++++++ Separate all the training data with index != 5 successfully! ++++++++++++++++++")

# separate all.dta with respect to index 1~4
for i in range(1, 5):
	print("++++++++++++++++++ Currently sperate index == " + str(i) + " from all.dta ... ++++++++++++++++++")
	train_idx = df_idx[df_idx.set == i].index.values

	df_train_with_date = df_all_with_date.loc[train_idx, :] # user, movie, date, rating
	df_train_with_date_arr = df_train_with_date.values

	df_train_no_date = df_all_no_date.loc[train_idx, :]
	df_train_no_date_arr = df_train_no_date.values

	#np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_" + str(i) + "_with_date.dta", df_train_with_date_arr, delimiter="\t", fmt='%10d')
	np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_" + str(i) + "_with_date.txt", df_train_with_date_arr, delimiter="\t", fmt='%10d')
	#np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_" + str(i) + "_no_date.dta", df_train_no_date_arr, delimiter="\t", fmt='%10d')
	np.savetxt("/Volumes/Disk1/ywwu/courses/cs156b/pre_processing/data/train_data_" + str(i) + "_no_date.txt", df_train_no_date_arr, delimiter="\t", fmt='%10d')
	print("++++++++++++++++++ Sperate index == " + str(i) + " from all.dta successfully! ++++++++++++++++++ ")

