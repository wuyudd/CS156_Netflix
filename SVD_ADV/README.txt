**************************************************************************
This folder contains the codes (main.cpp, SVD.cpp, SVD.h) for advanced 
SVD, which is with bias Bu (user) and Bi (movie) in regularization term.
**************************************************************************

**************************************************************************
HOW TO RUN THIS CODE:
**************************************************************************
To run this code, you need to: 
	1. Upload the data (train.txt, test.txt and probe.txt).
			-Format of train.txt, probe.txt: 
				userId movieId date rating
			-Format of test.txt: 
				userId movieId date
	2. Create a directory named "predictions" to save the predictions.
			-It will save the predictions every 5 iterations.
	3. Create a directory named "build" to cmake and make.
	4. Run ./executable_file_name (here: ./svd) in the "build" directory.

**************************************************************************
YOU CAN TUNE THE PARAMETERS!
**************************************************************************
The current parameters are set according to Koren's paper:
	"Koren_Advances_in_Collaborative_Filtering"
You can tune the parameters:
	-learning rate: learningrate
	-regularization: regularization
	-number of factors K: numFactor
	-number of iterations: maxIter (mostly, it converges at about 50 epochs)