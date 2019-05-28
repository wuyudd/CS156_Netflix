**************************************************************************
Environment:
**************************************************************************
Python:
	python 3.6
Package:
		numpy
		sklearn
		keras
		xgboost

**************************************************************************
FILE DESCRIPTION:
The .py files in this folder is to do the blending.
In the blending process:
	Training data: predictions of probe data
		-predictor is based on the training on index=1,2,3
		-format: userId movieId date rating
	Testing data: qual data
		-format: userId movieId date
**************************************************************************
linear_blending.py:
	Ridge Regression
	
	-input: 
		"input_args_path" includes both the paths of probe predictions 
	and the to-be-predicted qual data. One example is the "input.txt" in this
	folder. Attention: you need to put in all the .txt prediction files 
	mentioned in the "input.txt".
		"y_path" is the probe data path.
		"output_path" is the output file path.
	-output: a .txt file with predictions of the qual data.
	-run: python linear_blending.py input.txt probe.txt output.txt

nn_blending.py:
	Neural Network
		-two dense hidden layers with 64 units for each layer,
			"relu" activation
		-SGD optimizer, learning_rate = 5e-4, decay = 5e-7
	
	-input is the same as above.
	-output is the same as above.
	-run: python nn_blending.py input.txt probe.txt output.txt

xgboost_blend.py: 
	XGBoost Regressor
		-max_depth=6, n_estimators=70, other params set default.
		-you can still tune the parameters, it still has space for improvement.

	-input is the same as above.
	-output is the same as above.
	-run: python nn_blending.py input.txt probe.txt output.txt

get_mean.py:
	Get a mean prediction of the blended predictions.

	-input:
		"input_args_path" includes all the paths of the predictions you want to "mean".
		One example is the "input_mean.txt" in this folder. Attention: you need to put
		in all the .txt prediction files mentioned in the "input_mean.txt".
		"output_path" is the output file path.
	-output is the same as above.
	-run: python get_mean.py input.txt output.txt