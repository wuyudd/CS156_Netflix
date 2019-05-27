#include <unordered_map>
#include <tuple>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

class SparseMatrix{
public:
	std::vector<long> size;
	std::unordered_map<long, std::unordered_map<int, std::vector<double> > > matrix; //general sparse matrix representation
	std::unordered_map<int, std::vector<long> > movies_info; //for initialization of hidden bias only
	std::unordered_map<int, long> movies_num_ratings; //for initialization of hidden bias only

	SparseMatrix(long, int, int, std::string);
	void addData(std::string);
	bool ifUserWatchMovie(long, int);
	std::unordered_map<int, std::vector<double> > getWatchedMovie(long);
	int getRating(long,int);
	long getSize() {return matrix.size();}
};