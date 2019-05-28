**************************************************************************
This folder contains the codes (main.cpp, knn.cpp, knn.h) for movie-movie
K-Nearest Neighbors (KNN) with pearson correlation coefficients as metric.
**************************************************************************

**************************************************************************
HOW TO RUN THIS CODE:
**************************************************************************
To run this code, you need to: 
	1. Upload the data (train.txt, test.txt).
			-Format of train.txt: 
				userId movieId date rating
			-Format of test.txt: 
				userId movieId date
	2. Create a directory named "predictions" to save the predictions.
	3. Create a directory named "build" to cmake and make.
	4. Run ./executable_file_name (here: ./knn) in the "build" directory.

**************************************************************************
YOU CAN TUNE THE PARAMETERS!
**************************************************************************
You can tune the parameters:
	-number of nearest neighbors: K.