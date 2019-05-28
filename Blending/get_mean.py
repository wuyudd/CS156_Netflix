import sys
import numpy as np

def load_data(input_args_path):

    all_pred_paths = list() # all prediction file paths
    X = list()

    f_args = open(input_args_path, 'r')
    for line in f_args.readlines():
    	all_pred_paths.append(line.strip())
    f_args.close()

    for path in all_pred_paths:
        f = open(path, 'r')
        temp = list();
        for line in f.readlines():
            temp.append(float(line))
        X.append(temp)
        f.close()


    X_arr = np.asarray(X).T
    print("X_arr shape = ", X_arr.shape)

    return X_arr

if __name__ == '__main__':
    # read in the input arguments
    
    input_args_path = sys.argv[1] # probe and qual data for linear regression
    output_path = sys.argv[2]

    X_arr = load_data(input_args_path)
    X_mean_arr = np.mean(X_arr, axis=1)

    # X_final_arr = np.zeros(X_mean_arr.shape)

    # for i in range(len(X_mean_arr)):
    #     curr_val = X_mean_arr[i]
    #     if (curr_val - np.floor(curr_val) >= 0.95):
    #         X_final_arr[i] = np.ceil(curr_val)
    #     elif (curr_val - np.floor(curr_val) <= 0.05):
    #         X_final_arr[i] = np.floor(curr_val)
    #     else:
    #         X_final_arr[i] = curr_val

    # print("X_arr shape = ", X_arr.shape)
    # print("X_final_arr shape = ", X_final_arr.shape)
    # np.savetxt(output_path, X_final_arr)

    
    print("X_arr shape = ", X_arr.shape)
    print("X_mean_arr shape = ", X_mean_arr.shape)
    np.savetxt(output_path, X_mean_arr)
