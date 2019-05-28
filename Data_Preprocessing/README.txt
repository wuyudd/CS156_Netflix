**************************************************************************
Environment:
**************************************************************************
Python:
	python 3.6
Package:
		numpy
		pandas

**************************************************************************
FILE DESCRIPTION:
**************************************************************************
get_mu.py: 
	Calculate the mean rating of each subset of training data
		-index = 1, 2, 3, 4, (!=5), (!=4 && !=5)

find_max.py:
	Obtain the maximum user id and movie id in all the given data

sep_data.py: 
	Seprate the data according to index (index = 1, 2, 3, 4, (!=5, training data to predict qual data), (!=4 && !=5, training data to predict probe data)), save as .txt.
