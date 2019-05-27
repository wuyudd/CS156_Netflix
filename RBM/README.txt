********************************************************************************
DATA PREPARATION
********************************************************************************
data_spliter.py is for categorization of data: it splits original data into 5 files (etc. base.txt, hidden.txt, valid.txt, probe.txt, qual.txt) with no overlapping.

Run data_spliter:
python data_spliter.py 

********************************************************************************
RBM TRAINNING
********************************************************************************
Note that main.cpp is for learning starting from random initialization
		  main_transfer.cpp is for learning starting from pre-trained model

Compile:
g++ -w main.cpp SparseMatrix.cpp RBM.cpp -o executable_file_name
g++ -w main_transfer.cpp SparseMatrix.cpp RBM.cpp -o executable_file_name

Compile with compiler optimization:
g++ -w -O2 main.cpp SparseMatrix.cpp RBM.cpp -o executable_file_name
g++ -w -O2 main_transfer.cpp SparseMatrix.cpp RBM.cpp -o executable_file_name

Compiling with compiler optimization will have 7x speed improvement in training.

Run:
./executable_file_name

You can adjust several training parameters in main.cpp and main_transfer.cpp:
		learning_rate 
		weight_decay 
		epochs
		k (steps of gibbs sampling)
		num_hidden (number hidden units in RBM)

You can also change the directory path in both file for reading and writing of files.

		
