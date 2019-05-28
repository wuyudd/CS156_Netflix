# Created by Yu Wu.

import sys
import numpy as np
from xgboost import XGBRegressor
from sklearn import model_selection, metrics

def train_model(clf, X, Y): # validation, tune params
    kf = model_selection.KFold(n_splits=5)
    mse_train = []
    mse_valid = []
    for train_index, valid_index in kf.split(X):
#         print("TRAIN:", train_index, "VALIDATION:", valid_index)
        X_train, X_valid = X[train_index], X[valid_index]
        Y_train, Y_valid = Y[train_index], Y[valid_index]
#         print(X_train.shape)
#         print(Y_train.shape)
        clf.fit(X_train, Y_train)
        Y_train_pred = clf.predict(X_train)
        Y_valid_pred = clf.predict(X_valid)
		
        mse_train_sub = metrics.mean_squared_error(Y_train, Y_train_pred)
        mse_valid_sub = metrics.mean_squared_error(Y_valid, Y_valid_pred)
		
        print("train mse: ", mse_train_sub, "valid mse: ", mse_valid_sub)
        mse_train.append(mse_train_sub)
        mse_valid.append(mse_valid_sub)
        return mse_train, mse_valid

def load_data(input_args_path, y_path, output_path):
    X_train = list() # train data paths
    X_test = list() # to-predict test data paths
    X = list() # probe X train data
    y = list() # probe true y
    X_qual = list() # qual X_qual predictions data

    f_args = open(input_args_path, 'r')
    for line in f_args.readlines():
        [train, test] = line.split()
        # print(train)
        # print(test)
        X_train.append(train)
        X_test.append(test)
    f_args.close()

    f_y = open(y_path, 'r')
    for line in f_y.readlines():
        label = float(line.split()[-1])
        y.append(label)
    f_y.close()
        
    for path in X_train:
        # print("xtrain")
        # print(path)
        f = open(path, 'r')
        temp = list();
        for line in f.readlines():
            temp.append(float(line))
        # print("temp_shape=", len(temp))
        X.append(temp)
        f.close()


    for path in X_test:
        # print("xtest")
        # print(path)
        f = open(path, 'r')
        temp = list();
        for line in f.readlines():
            temp.append(float(line))
        # print("temp_shape=", len(temp))
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
    
    input_args_path = sys.argv[1] # probe and qual data for linear regression
    y_path = sys.argv[2] # probe true y
    output_path = sys.argv[3]

    X_arr, y_arr, X_qual_arr = load_data(input_args_path, y_path, output_path)
    #clf_whole = XGBRegressor(booster='gbtree', objective='reg:linear', gamma=0.1, max_depth=10, reg_lambda=3, subsample=0.7, colsample_bytree=0.7, min_child_weight=3, silent=1, eta=0.007, seed=1000, nthread=4)
    #clf_whole = XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=80, min_child_weight=1, subsample=0.85)
    clf_whole = XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=70, min_child_weight=1)
    clf_whole.fit(X_arr, y_arr)
    blended_rating = clf_whole.predict(X_qual_arr)
    print(blended_rating)
    np.savetxt(output_path, blended_rating)