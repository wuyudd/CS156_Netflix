**************************************************************************
This folder contains the codes (main.cpp, time_svd_pp.cpp, time_svd_pp.hpp) 
for time SVD++.
**************************************************************************

**************************************************************************
HOW TO RUN THIS CODE:
**************************************************************************
To run this code, you need to: 
	1. Upload the data (train.txt, valid.txt, test.txt and probe.txt).
			-Format of train.txt, valid.txt, probe.txt: 
				userId movieId date rating
			-Format of test.txt: 
				userId movieId date
	2. Create a directory named "predictions" to save the predictions.
			-It will save the predictions every 5 iterations.
	3. Create a directory named "build" to cmake and make.
	4. Run ./executable_file_name (here: ./time_svd) in the "build" directory.
