# Created by Yu Wu.

from keras.datasets import boston_housing
from keras import models, layers, optimizers
import numpy as np
import sys
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

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #model.add(layers.Activation('tanh'))
    sgd = optimizers.SGD(lr=5e-4, decay=5e-7)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    input_args_path = sys.argv[1] # probe and qual data for linear regression
    y_path = sys.argv[2] # probe true y
    output_path = sys.argv[3] # the output prediction path

    X_train, y_train, X_test = load_data(input_args_path, y_path, output_path)
    num_epochs = 1000
    model = build_model()
    model.fit(X_train, y_train,epochs=num_epochs, batch_size=128, verbose=1)
    predicts = model.predict(X_test)
    print(predicts)
    np.savetxt(output_path, predicts)

    # num_ratings = blended_rating_ori.shape[0]
    # print("size of blended_rating: ", num_ratings)
    # alpha = 3.6
    # beta = 3.0
    # blended_rating = np.zeros(num_ratings)
    # for i in range(num_ratings):
    # 	blended_rating[i] = blended_rating_ori[i] * alpha + beta
    # print("............ blended_rating ............")
    # print(blended_rating)
    # np.savetxt(output_path, blended_rating)