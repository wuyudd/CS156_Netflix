//
//  time_svd_pp.cpp
//  time_svd_pp
//
//  Created by Diyi Liu.
//  Copyright © 2019 Diyi Liu. All rights reserved.
//
#include "time_svd_pp.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

static const double PI = 3.141592654;
static const int USER_NUM = 458294;                   // number of users + 1
static const int MOVIE_NUM = 17771;                   // number of movies + 1
static const int DATA_NUM = 2244;                     // number of date + 1
static const double RATING_AVG = 3.6095161972728063;  // average score

void TimeSVDpp::Learning() {
    double cur_ein = 0.0;
    cout << "********** Starting training with M = " << USER_NUM << " users, N = " << MOVIE_NUM << " items. **********" << endl;
    cout << "********** Training with Factor K = " << factor << ", Max Iteration = " << max_iter << ". **********" << endl;
    for(size_t i=0; i<max_iter; i++) {
        Train();
        cur_ein = InError();
        cout << "In-sample error in iteration " << i << ": " << cur_ein << endl;
        if ((i + 1) % 5 == 0) {
            SaveTemp(i);
        }
    }
    cout << "Final in-sample RMSE = " << cur_ein << endl;
    cout << "Begin to write final prediction to file..." << endl;
    ofstream fout(qual_out_file.c_str());
    ifstream ftest(qual_file, ios::in);
    int user, movie, date;
    while (ftest >> user >> movie >> date) {
        fout << Predict(user, movie, date) << endl;
    }
    //    if(!ftest.eof()) runtime_error("Invalid data from qual file!");
    ftest.close();
    fout.close();
    //PredictProbe();
    cout << "Success!" << endl;
}

void TimeSVDpp::SaveTemp(size_t index) {
    int user, movie, date;
    ostringstream oss2;
    oss2 << "../predictions/time_svd_pp_no45_probe_iter_" << index+1 << ".txt";
    string out_temp2 = oss2.str();
    ofstream f_probe_out_inter(out_temp2.c_str());
    ifstream fin_probe_inter(probe_file, ios::in);
    while (fin_probe_inter >> user >> movie >> date) {
        f_probe_out_inter << Predict(user, movie, date) << endl;
    }
    //    if(!f_probe_out_inter.eof()) throw runtime_error("Invalid data from probe file!");
    fin_probe_inter.close();
    f_probe_out_inter.close();
    ostringstream oss3;
    oss3 << "../predictions/time_svd_pp_no45_qual_iter_" << index+1 << ".txt";
    string out_temp3 = oss3.str();
    ofstream f_qual_out_inter(out_temp3.c_str());
    ifstream fin_qual_inter(qual_file, ios::in);
    while (fin_qual_inter >> user >> movie >> date) {
        f_qual_out_inter << Predict(user, movie, date) << endl;
    }
    //    if(!fin_qual_inter.eof()) throw runtime_error("Invalid data from qual file!");
    fin_qual_inter.close();
    f_qual_out_inter.close();
    
}

// in-sample error
double TimeSVDpp::InError() {
    int user, movie, rating, date;
    int count = 0;
    double squre_error = 0.0;
    for (int p = 0; p < train_data.size(); ++p) {
        user = train_data[p].user;
        movie = train_data[p].movie;
        date = train_data[p].date;
        rating = train_data[p].rating;
        ++count;
        double diff_rating = rating - Predict(user, movie, date);
        squre_error += diff_rating * diff_rating;
    }
    return sqrt(squre_error / count);
}

// predict rating for (user, movie, date)
double TimeSVDpp::Predict(int user, int movie,int date){
    double temp = 0.0;
    double num_movies = user_num_movies[user];
    double sqrt_num_inv = (num_movies >= 1) ? (1 / (sqrt(num_movies))) : 0.0;
    for(size_t i=0; i<factor; i++){
        temp += (Pu[user][i] +sum_Pu[user][i]*sqrt_num_inv) * Qi[movie][i];
    }
    double pred_rating = RATING_AVG + Bu[user] + Bi[movie] + Bi_bin[movie][DateBin(date)] + alpha_u[user]*DevUt(user, date) + Bu_t[user][date] + temp;
    if(pred_rating > 5.0)
        pred_rating = 5.0;
    if(pred_rating < 1.0)
        pred_rating = 1.0;
    return pred_rating;
}

// error on probe data
void TimeSVDpp::PredictProbe(){
    ofstream fout(probe_out_file.c_str());
    ifstream fprobe(probe_file, ios::in);
    int user, movie, date;
    while (fprobe >> user >> movie >> date) {
        fout << Predict(user, movie, date) << endl;
    }
//    if(!fout.eof()) throw runtime_error("Invalid data from probe file");
    fprobe.close();
}

void TimeSVDpp::Train(){
    int user,movie,rating,date;
    int pre_user = 0, num_movies = 0, end = 0;
    double sqrt_num_inv = 0;

    vector <double> temp_sum(factor,0);
    for (int i = 0; i < train_data.size(); ++i) {
        user = train_data[i].user;
        movie = train_data[i].movie;
        rating = train_data[i].rating;
        date = train_data[i].date;
        // new user appears
        if (user != pre_user) {
            // initialize temp_sum
            for (int k = 0; k < factor; ++k) {
                temp_sum.at(k) = 0;
            }
            num_movies = user_num_movies[user];
            sqrt_num_inv = (num_movies >= 1) ? (1 / (sqrt(num_movies))) : 0.0; // ||Ru||^(-1/2)
            // update sum_Pu
            for (int k = 0; k < factor; ++k) {
                int curr_movie = 0;
                double sumy = 0.0;
                for (int p = i; p < i + num_movies; ++p) {
                    curr_movie = train_data[p].movie;
                    sumy += y[curr_movie][k];
                }
                sum_Pu[user][k] = sumy;
            }
            end = i + num_movies - 1;
        }
        // update per data point
        double diff_rating = rating - Predict(user,movie,date);
        Bu[user] += lr_init * (diff_rating - la_bias * Bu[user]);
        Bi[movie] += lr_init * (diff_rating - la_bias * Bi[movie]);
        Bi_bin[movie][DateBin(date)] += lr_init * (diff_rating - la_bias * Bi_bin[movie][DateBin(date)]);
        alpha_u[user] += lr_alpha * (diff_rating * DevUt(user, date)  - la_alpha * alpha_u[user]);
        Bu_t[user][date] += lr_init * (diff_rating - la_bias * Bu_t[user][date]);
        for(int k=0; k<factor; k++){
            double pu_val = Pu[user][k];
            double qi_val = Qi[movie][k];
            Pu[user][k] += lr_init * (diff_rating * qi_val - la_PQY * pu_val);
            Qi[movie][k] += lr_init * (diff_rating * (pu_val+sqrt_num_inv*sum_Pu[user][k]) - la_PQY * qi_val);
            temp_sum[k] += diff_rating * sqrt_num_inv * qi_val;
        }

        if (i == end) {
            // update y per user, approximation
            for (int k = 0; k < factor; ++k) {
                int curr_movie = 0;
                for (int p = i - user_num_movies[user] + 1; p < i + 1; ++p) {
                    curr_movie = train_data[p].movie;
                    double temp = y[curr_movie][k];
                    y[curr_movie][k] += lr_init * (temp_sum[k] - la_PQY * temp);
                    sum_Pu[user][k] += y[curr_movie][k] - temp;
                }
            }
        }
        pre_user = user;
    }
    lr_init *= lr_decay;
    lr_alpha *= lr_decay;
}

TimeSVDpp::TimeSVDpp(const string& train_file, const string& validation_file, const string& qual_file, const string& qual_out_file, const string& probe_file, const string& probe_out_file, double lr_init, double lr_alpha, double lr_decay, double la_bias, double la_PQY, double la_alpha,double threshold, int bin_num, int factor, int max_iter):
    train_file(train_file),
    validation_file(validation_file),
    qual_file(qual_file),
    qual_out_file(qual_out_file),
    probe_file(probe_file),
    probe_out_file(probe_out_file),
    lr_init(lr_init),
    lr_alpha(lr_alpha),
    lr_decay(lr_decay),
    la_bias(la_bias),
    la_PQY(la_PQY),
    la_alpha(la_alpha),
    threshold(threshold),
    bin_num(bin_num),
    factor(factor),
    max_iter(max_iter),
    Bi(MOVIE_NUM, 0.0),
    Bu(USER_NUM, 0.0),
    Pu(USER_NUM, vector<double>(factor, 0.0)),
    Qi(MOVIE_NUM, vector<double>(factor, 0.0)),
    y(MOVIE_NUM, vector<double>(factor, 0.0)),
    Bi_bin(MOVIE_NUM, vector<double>(bin_num, 0.0)),
    alpha_u(USER_NUM, 0.0),
    sum_Pu(USER_NUM, vector<double>(factor, 0.0)),
    mu_Tu(USER_NUM, 0.0),
    user_num_movies(USER_NUM, 0){
        
    ifstream ftrain(train_file, ios::in);
    int user, movie, date, rating;
    int pre_user = 0; // previous userId
    int num_movies = 0;
    // load train data
    while (ftrain >> user >> movie >> date >> rating) {
        if (user != pre_user) {
            user_num_movies.at(pre_user) = num_movies; // save the number of movies the user rated
            num_movies = 0;
        }
        ++num_movies;
        
        Data dataPoint;
        dataPoint.user = user;
        dataPoint.movie = movie;
        dataPoint.date = date;
        dataPoint.rating = rating;
        train_data.push_back(dataPoint);
        pre_user = user;
    }
    user_num_movies.at(pre_user) = num_movies;
    if (!ftrain.eof()) throw runtime_error("Invalid data from trainFile!");
    ftrain.close();
//    cout << train_data.size() << endl;
    
    for(size_t i=0; i<MOVIE_NUM; i++){
        for(size_t j=0; j<factor; j++){
            double U1 = rand() / (RAND_MAX + 1.0);
            double V1 = rand() / (RAND_MAX + 1.0);
            Qi[i][j] = 0.1 * sqrt(-2.0 * log(U1))* sin(2.0 * PI * V1); // initialization to normal distribution with mu=0, sigma=0.1
        }
    }
    
    for(size_t i=0; i<USER_NUM; i++){
        for(size_t j=0; j<factor; j++){
            double U2 = rand() / (RAND_MAX + 1.0);
            double V2 = rand() / (RAND_MAX + 1.0);
            sum_Pu[i][j] = 0.1 * sqrt(-2.0 * log(U2))* cos(2.0 * PI * V2);
            Pu[i][j] = 0.1 * sqrt(-2.0 * log(U2))* sin(2.0 * PI * V2); // initialization to normal distribution with mu=0, sigma=0.1
        }
    }
    
    num_movies = 0;
    pre_user = 0;
    int end = 0;
    double temp = 0;
    for (int i = 0; i < train_data.size(); ++i) {
        user = train_data[i].user;
        movie = train_data[i].movie;
        date = train_data[i].date;
        rating = train_data[i].rating;
        
        // a new userId appears
        if (user != pre_user) {
            if (num_movies == 0) {
                mu_Tu[pre_user] = 0;
            }
            else {
                mu_Tu[pre_user] = temp / num_movies;
            }
//            cout << "mu_Tu[" << pre_user << "] = " << mu_Tu[pre_user] << endl;
            num_movies = user_num_movies[user];
//            cout << "num_movies[" << user << "] = " << num_movies << endl;
            temp = 0;
            pre_user = user;
            end = i + num_movies - 1;
        }
        temp += date;
        if (i == end) {
            mu_Tu[user] = temp / user_num_movies[user];
//            cout << "mu_Tu[" << pre_user << "] = " << mu_Tu[pre_user] << endl;
        }
    }
    
    pre_user = 0;
    end = 0;
    map<int,double> map_temp = {};
    for (int i = 0; i < train_data.size(); ++i) {
        user = train_data[i].user;
        date = train_data[i].date;
        if (user != pre_user) {
            Bu_t.push_back(map_temp);
            map_temp = {};
            int temp_user = pre_user + 1;
            while (temp_user != user) {
                Bu_t.push_back({});
                temp_user += 1;
            }
        }
        if (map_temp.count(date) == 0) {
            map_temp[date] = 0.0;
        }
        pre_user = user;
    }
    Bu_t.push_back(map_temp);
//    cout << Bu_t.size() << endl;
    
    for(size_t i=0; i<USER_NUM; i++){
        map<int,double> temp;
        Dev.push_back(temp);
    }
        
    cout << "Complete Construction!" << endl;
}

// compute and save DevUt
double TimeSVDpp::DevUt(int user, int date) {
    if(Dev[user].count(date) == 0) {
        double temp = (double)date - mu_Tu[user];
        double temp1 = 0.0;
        if (temp < 0) {
            temp1 = -1 * pow(abs(temp), 0.4);
        } else if (temp > 0) {
            temp1 = pow(abs(temp), 0.4);
        } else {
            temp1 = 0;
        }
        Dev[user][date] = temp1;
        return temp1;
    }
    else return Dev[user][date];
}

int TimeSVDpp::DateBin(int date) {
    int binsize = DATA_NUM/bin_num + 1;
    return date/binsize;
}
