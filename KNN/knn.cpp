#include "knn.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <queue>

using namespace std;

static const int USER_NUM = 458294;
static const int MOVIE_NUM = 17771;

// Main train function
void KNN::Query() {
    cout << "Start to calculate Pearson Correlation Matrix ..." << endl;
    PearsonCoeff();
    cout << "Calculated Pearson Correlation Matrix!" << endl;
    cout << "Start to Predict ..." << endl;
    SaveQual();
    cout << "Predictions Successfully saved!" << endl;
}

void KNN::SaveQual() {
    int userId, movieId, date;
    ofstream fOut(out_file.c_str());
    ifstream fTest(test_file, ios::in);
    while (fTest >> userId >> movieId >> date) {
        fOut << PredictRating(userId, movieId) << endl;
    }
    fTest.close();
    fOut.close();
}

// Calculate pearson correlation coefficient between every two movies
void KNN::PearsonCoeff() {
	// go through every movie (calculate each row of pearson matrix)
    for (int movie1 = 0; movie1 < MOVIE_NUM; ++movie1) {
        if ((movie1 % 1000) == 0) {
            cout << "Current movie1 = " << movie1 << endl;
        }
        int movie1_users_num = movie_user[movie1].size();
        vector<PearsonIntermediate> row_movie1(MOVIE_NUM);
        for (int movieId = 0; movieId < MOVIE_NUM; ++movieId) {
            row_movie1[movieId].i = 0;
            row_movie1[movieId].j = 0;
            row_movie1[movieId].ii = 0;
            row_movie1[movieId].jj = 0;
            row_movie1[movieId].ij = 0;
            row_movie1[movieId].cnt = 0;
        }
        int total_rating = 0;
        // go through every users rating movie1
        for (int idx_movie1 = 0; idx_movie1 < movie1_users_num; ++idx_movie1) {
            int user = movie_user[movie1][idx_movie1].first;
            int rating_movie1_user = movie_user[movie1][idx_movie1].second;
            int user_movies_num = user_movie[user].size();
            total_rating += rating_movie1_user;
            
			// go through every movie rated by the same user rating movie1
            for (int idx_user = 0; idx_user < user_movies_num; ++idx_user) {
                int movie2 = user_movie[user][idx_user].first;
                int rating_movie2_user = user_movie[user][idx_user].second;
                row_movie1[movie2].i += rating_movie1_user;
                row_movie1[movie2].j += rating_movie2_user;
                row_movie1[movie2].ii += rating_movie1_user * rating_movie1_user;
                row_movie1[movie2].jj += rating_movie2_user * rating_movie2_user;
                row_movie1[movie2].ij += rating_movie1_user * rating_movie2_user;
                row_movie1[movie2].cnt += 1;
            }
        }
        
        for (int movieId = 0; movieId < MOVIE_NUM; ++movieId) {
            PearsonIntermediate pi = row_movie1[movieId];
            float p;
            
            if (pi.cnt != 0) {
				// deal with invalid denominator
                if ((pi.cnt * pi.ii - pi.i * pi.i == 0) || (pi.cnt * pi.jj - pi.j * pi.j == 0)) {
                    p = 0.0;
                }else {
                    p = (pi.cnt * pi.ij - pi.i * pi.j) / (sqrt(pi.cnt * pi.ii - pi.i * pi.i)
                                                          * sqrt(pi.cnt * pi.jj - pi.j * pi.j));
                }
                if (isnan(p)) {
                    p = 0.0;
                }
                pearson_matrix[movie1][movieId].first = pi.cnt;
                pearson_matrix[movie1][movieId].second = p;
            }
        }
        if (movie1_users_num > 0){
            movie_mean[movie1] = (float)total_rating / movie1_users_num;
        }
    }
}

// Predict the rating the userId gives to movieId
double KNN::PredictRating(int user_id, int movie1){
    priority_queue<pair<float, int>> corr_movies;
	// default value
    // float default_value = AVG + (avg_user_raings[user_id] - AVG) * 1.0 / (1.0 + exp(-user_movie[user_id].size() / MOVIE_NUM))
    //         + (avg_movie_ratings[movie1] - AVG) * 1.0 / (1.0 + exp(-movie_user[movie1].size() / USER_NUM));
	
    // obtain all other movies rated by the same user
    int num_movies = user_movie[user_id].size();
    for (int m = 0; m < num_movies; ++m) {
        int movie2 = user_movie[user_id][m].first;
        float pearson_curr = pearson_matrix[movie1][movie2].second;
        corr_movies.push(make_pair(pearson_curr, m));
    }

    float numerator = 0.0;
    float denominator = 0.0;
    int corr_movies_size = corr_movies.size();
	// obtain k movies rated by the same user which are more similar to current movie (If less than k, obtain all)
	// predict the rating by weighted average.
    for (int k = 0; k < min(K, corr_movies_size); ++k) {
        float corr_val = corr_movies.top().first;
        int curr_pos = corr_movies.top().second;
        int corr_movie = user_movie[user_id][curr_pos].first;
        corr_movies.pop();
        numerator += corr_val * (user_movie[user_id][curr_pos].second - movie_mean[corr_movie]);
        denominator += abs(corr_val);
    }
    
    if (denominator == 0.0) {
		// If current user did not rate other movies, use the mean rating of current movie to estimate.
        return movie_mean[movie1];
    } else {
		// control the predicted rating in the range of [1, 5]
        if (numerator / denominator + movie_mean[movie1] < 1.0) {
            return 1.0;
        }
        else if (numerator / denominator + movie_mean[movie1] > 5.0) {
            return 5.0;
        }
        else {
            return numerator / denominator + movie_mean[movie1];
        }
    }
}

KNN::KNN(const string& train_file, const string& test_file, const string& out_file, int K) :
train_file(train_file), test_file(test_file), out_file(out_file), K(K),
pearson_matrix(MOVIE_NUM, vector<pair<int, float>>(MOVIE_NUM, make_pair(0,0.0))),
movie_mean(MOVIE_NUM, 0.0),
avg_user_raings(USER_NUM, 0.0), avg_movie_ratings(MOVIE_NUM, 0.0) {
    user_movie.resize(USER_NUM);
    movie_user.resize(MOVIE_NUM);
    ifstream fTrain(train_file, ios::in);
    int userId, movieId, date, rating;
    // load train data
    while (fTrain >> userId >> movieId >> date >> rating) {
        movie_user.at(movieId).push_back(make_pair(userId, rating));
        user_movie.at(userId).push_back(make_pair(movieId, rating));
        avg_user_raings[userId] += rating;
        avg_movie_ratings[movieId] += rating;
    }
    fTrain.close();
    
	// calculate every movie's mean rating and every user's mean rating
    for (int u = 0; u < USER_NUM; ++u) {
        avg_user_raings[u] /= user_movie[u].size();
    }
    for (int m = 0; m < MOVIE_NUM; ++m) {
        avg_movie_ratings[m] /= movie_user[m].size();
    }
}


