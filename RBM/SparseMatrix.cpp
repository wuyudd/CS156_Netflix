#include "SparseMatrix.h"
// #include <tuple>

//load data from txt file to sparse matrix representation
SparseMatrix::SparseMatrix(long num_users, int num_movies, int num_ratings, std::string file_name){
	size.push_back(num_users);
	size.push_back(num_movies);
	size.push_back(num_ratings);
	std::ifstream file;
	file.open(file_name);
	if (not file.is_open()){
		std::cout << "Unable to open the data file,check the directory for file" << std::endl;
		exit(1);
	}
	std::cout << "Start Loading Trainning Data" << std::endl;
	std::string line;
	while (!file.eof()){
		getline(file, line);
		if(line == ""){
            break;
        }
		std::istringstream lineis(line);
		long user;
		int movie, date, rating;
		lineis >> user >> movie >> date >> rating;
		user -= 1;
		movie -= 1;
		rating -= 1;
		std::vector<double> temp(num_ratings,0.0);
		matrix[user][movie] = temp;
		matrix[user][movie][rating] = 1.0;

		if (movies_info.find(movie) == movies_info.end()){
			std::vector<long> temp(num_ratings, 0);
			movies_info[movie] = temp;
			movies_num_ratings[movie] = 0;
		}
		movies_info[movie][rating] += 1;
		movies_num_ratings[movie] += 1;
	}
	std::cout << "Complete Loading Trainning Data" << std::endl;
}

//add data to current sparse matrix
void SparseMatrix::addData(std::string file_name){
	std::ifstream file;
	file.open(file_name);
	if (not file.is_open()){
		std::cout << "Unable to open the data file,check the directory for file" << std::endl;
		exit(1);
	}
	std::cout << "Start Adding " << file_name  << " to training data"<<std::endl;
	std::string line;
	while (!file.eof()){
		getline(file, line);
		if(line == ""){
            break;
        }
		std::istringstream lineis(line);
		long user;
		int movie, date, rating;
		lineis >> user >> movie >> date >> rating;
		user -= 1;
		movie -= 1;
		rating -= 1;
		std::vector<double> temp(size[2],0.0);
		matrix[user][movie] = temp;
		matrix[user][movie][rating] = 1.0;
	
		if (movies_info.find(movie) == movies_info.end()){
			std::vector<long> temp(size[2], 0);
			movies_info[movie] = temp;
			movies_num_ratings[movie] = 0;
		}
		movies_info[movie][rating] += 1;
		movies_num_ratings[movie] += 1;
	}
	std::cout << "Complete Adding " << file_name  << " to training data"<<std::endl;
}


bool SparseMatrix::ifUserWatchMovie(long user,int movie){
	if (matrix.find(user) == matrix.end()){
		std::cout << "user not in matrix" << std::endl;
		exit(1);
	}
	return matrix[user].find(movie) != matrix[user].end();
}

std::unordered_map<int, std::vector<double> > SparseMatrix::getWatchedMovie(long user){
	// if (matrix.find(user) == matrix.end()){
	// 	std::cout << "user not in data" << std::endl;
	// 	exit(1);
	// }
	return matrix[user];
}

int SparseMatrix::getRating(long user,int movie){
	if (matrix.find(user) == matrix.end()){
		std::cout << "user not in data" << std::endl;
		exit(1);
	}
	if (matrix[user].find(movie) == matrix[user].end()){
		std::cout << "movie has not been rated by user" << std::endl;
		exit(1);
	}
	int length = size[2];
	double rating = 0.0;
	for(int i = 0; i < length ;i++){
		if (matrix[user][movie][i] == 1.0){
			rating = i + 1.0;
			break;
		}
	}
	return rating;
}
