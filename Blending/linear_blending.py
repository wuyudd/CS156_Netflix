# Created by Yu Wu.

import sys
import numpy as np
from sklearn import linear_model

def load_data(input_args_path, y_path, output_path):
	X_train = list() # train data paths
	X_test = list() # to-predict test data paths
	X = list() # probe X train data
	y = list() # probe true y
	X_qual = list() # qual X_qual predictions data

	f_args = open(input_args_path, 'r')
	for line in f_args.readlines():
		[train, test] = line.split()
		X_train.append(train)
		X_test.append(test)
	f_args.close()

	f_y = open(y_path, 'r')
	for line in f_y.readlines():
		label = float(line.split()[-1])
		y.append(label)
	f_y.close()
		
	for path in X_train:
		f = open(path, 'r')
		temp = list();
		for line in f.readlines():
			temp.append(float(line))
		X.append(temp)
		f.close()


	for path in X_test:
		f = open(path, 'r')
		temp = list();
		for line in f.readlines():
			temp.append(float(line))
		X_qual.append(temp)
		f.close()

	X_arr = np.asarray(X).T
	print("X_arr shape = ", X_arr.shape)

	y_arr = np.asarray(y).T
	print("y_arr shape = ", y_arr.shape)

	X_qual_arr = np.asarray(X_qual).T
	print("X_qual_arr shape = ", X_qual_arr.shape)

	return X_arr, y_arr, X_qual_arr

if __name__ == '__main__':
	# read in the input arguments
	input_args_path = sys.argv[1] # probe and qual data (paths) for linear regression
	y_path = sys.argv[2] # probe true y
	output_path = sys.argv[3] # the output prediction path
	X_arr, y_arr, X_qual_arr = load_data(input_args_path, y_path, output_path)

	reg = linear_model.Ridge(alpha=5e-6) # ridge regression
	reg.fit(X_arr, y_arr) # train
	weights = reg.coef_
	bias = reg.intercept_
	print("++++++++++++++ WEIGHTS ++++++++++++++")
	print(weights)
	print("++++++++++++++ BIAS ++++++++++++++")
	print(bias)
	
	blended_rating = reg.predict(X_qual_arr) # predict
	print("++++++++++++++ FINAL RATINGS! ++++++++++++++")
	print(blended_rating)
	np.savetxt(output_path, blended_rating)










