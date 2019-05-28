// Created by Yu Wu

#include "SVD.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

static const double PI = 3.141592654;
static const int USER_NUM = 458294;
static const int MOVIE_NUM = 17771;
static const double RATING_AVG = 3.6095161972728063; // mu, index != 5

// save temp results
void SVD::SaveTemp(int index) {
    int user, movie, date;
    ostringstream ossInter;
    ossInter << "../predictions/inter_svdpp_qual_K-" << numFactor << "_Iter-" << index + 1 << ".txt";
    ofstream fOutInter(ossInter.str()); // intermediate output
    ifstream fTestInter(testFile, ios::in);
    while (fTestInter >> user >> movie >> date) {
        fOutInter << PredictRating(user, movie) << endl;
    }
    if (!fTestInter.eof()) throw runtime_error("Invalid data from file!");
    fTestInter.close();
    fOutInter.close();
}

// save qual results
void SVD::SaveQual() {
    int user, movie, date;
    ofstream fOut(outFile.c_str());
    ifstream fTest(testFile, ios::in);
    while (fTest >> user >> movie >> date) {
        fOut << PredictRating(user, movie) << endl;
    }
    if (!fTest.eof()) throw runtime_error("Invalid data from file!");
    fTest.close();
    fOut.close();
}

double SVD::Train() {
    srand((unsigned)time(nullptr));
    double currRmse = 0.0;
    cout << "********** Starting training with M = " << USER_NUM << " users, N = " << MOVIE_NUM << " items. **********" << endl;
    cout << "********** Training with Factor K = " << numFactor << ", Max Iteration = " << maxIter << ". **********" << endl;
    for(int i = 0; i < maxIter; ++i) {
        cout << "Current Iteration = " << i + 1 << endl;
        TrainInternal(); // update
        currRmse = ErrorIn(); // in-sample error
        // double currRmseProbe = ErrorOut();
        cout << "       Now in-sample error at iteration " << i + 1 << ": " << currRmse << endl;
        // save the intermediate result
        if ((i + 1) % 5 == 0) {
            SaveTemp(i);
        }
    }
    cout << "Currently doing predictions on test ..." << endl;
    SaveQual();
    cout << "Predictions done!" << endl;
    return currRmse;
}

// in-sample error
double SVD::ErrorIn() {
    int userId, movieId, rating;
    int count = 0;
    double squErr = 0;
    for (int p = 0; p < trainData.size(); ++p) {
        userId = trainData[p].userId;
        movieId = trainData[p].movieId;
        rating = trainData[p].rating;
        ++count;
        double diffRating = rating - PredictRating(userId, movieId);
        squErr += diffRating * diffRating;
    }

    return sqrt(squErr / count); // mse
}

// out-sample error on probe data
double SVD::ErrorOut() {
    int userId, movieId, date, rating;
    int count = 0;
    double squErr = 0.0;
    ifstream fProbe(probeFile, ios::in);
    while (fProbe >> userId >> movieId >> date >> rating) {
        double diffRating = rating - PredictRating(userId, movieId);
        squErr += diffRating * diffRating;
        ++count;
    }
    if (!fProbe.eof()) throw runtime_error("Invalid data from file!");
    fProbe.close();
    //cout << "       ErrorOut function: count = " << count << endl;
    return sqrt(squErr / count);
}

// predict the rating the userId gives to movieId
double SVD::PredictRating(int userId, int movieId){
    int numMovie = userNumMovies[userId];
    double sqrtNumMovInv = (numMovie >= 1) ? (1/(sqrt(numMovie))) : 0.0; // ||Ru||^(-1/2)
    double tempRes = 0.0;
    for(int k = 0; k < numFactor; ++k) {
        tempRes += (Pu[userId][k] + sqrtNumMovInv * sumYj[userId][k]) * Qi[movieId][k];
    }
    double predRating = RATING_AVG + Bu[userId] + Bi[movieId] + tempRes; // predicted rating (userId, movieId)
    // control the predicted rating in the range of [1, 5]
    if(predRating > 5.0){
        predRating = 5.0;
    } else if(predRating < 1.0){
        predRating = 1.0;
    }
    return predRating;
}

void SVD::TrainInternal() {
    int userId, movieId, rating;
    int preUserId = 0;
    int numMovie = 0;
    double sqrtNumMovInv = 0.0;
    int end = 0;
    vector<double> tempYPart(numFactor, 0);
    // initialize sumYi for every iteration
    for (int u = 0; u < USER_NUM; ++u) {
        for (int k = 0; k < numFactor; ++k) {
            sumYj[u][k] = 0.0;
        }
    }

    // go through every data point
    for (int i = 0; i < trainData.size(); ++i) {
        userId = trainData[i].userId;
        movieId = trainData[i].movieId;
        rating = trainData[i].rating;

        // a new userId appears
        if (userId != preUserId) {
            // for a new user, initialize a new tempYPart with value 0
            for (int k = 0; k < numFactor; ++k) {
                tempYPart.at(k) = 0;
            }
            numMovie = userNumMovies[userId];
            sqrtNumMovInv = (numMovie >= 1) ? (1 / (sqrt(numMovie))) : 0.0; // ||Ru||^(-1/2)
            // update sumYj for a user batch
            for (int k = 0; k < numFactor; ++k) {
                int currMovieId = 0;
                for (int p = i; p < i + numMovie; ++p) {
                    currMovieId = trainData[p].movieId;
                    sumYj[userId][k] += y[currMovieId][k];
                }
            }
            end = i + numMovie - 1;
        }

        // update Bu, Bi, Pu, Qi per data point
        double diffRating = rating - PredictRating(userId, movieId);
        Bu[userId] += lrInit * (diffRating - laBias * Bu[userId]);
        Bi[movieId] += lrInit * (diffRating - laBias * Bi[movieId]);
        for (int k = 0; k < numFactor; ++k) {
            double puVal = Pu[userId][k];
            double qiVal = Qi[movieId][k];
            Pu[userId][k] += lrInit * (diffRating * qiVal - laFactor * puVal);
            Qi[movieId][k] += lrInit * (diffRating * (puVal + sqrtNumMovInv * sumYj[userId][k]) - laFactor * qiVal);
            tempYPart[k] += diffRating * sqrtNumMovInv * qiVal; // save part of y value for later update of y
        }

        if (i == end) {
            // update y per user
            for (int k = 0; k < numFactor; ++k) {
                int currMovieId = 0;
                for (int p = i - numMovie + 1; p < i + 1; ++p) {
                    currMovieId = trainData[p].movieId;
                    double yjVal = y[currMovieId][k];
                    y[currMovieId][k] += lrInit * (tempYPart[k] - laFactor * yjVal);
                    sumYj[userId][k] += (y[currMovieId][k] - yjVal);
                }
            }
        }

        preUserId = userId;
    }
    // learning rate decay = 0.9
    lrInit *= lrDecay;
}


SVD::SVD(const string &trainFile, const string &testFile, const string &probeFile, const string &outFile, double lrInit, double lrDecay,
         double laBias, double laFactor, int numFactor, int maxIter) :
        trainFile(trainFile),testFile(testFile), probeFile(probeFile), outFile(outFile),
        lrInit(lrInit), lrDecay(lrDecay),
        laBias(laBias), laFactor(laFactor),
        numFactor(numFactor), maxIter(maxIter),
        Bu(USER_NUM, 0.0),
        Bi(MOVIE_NUM, 0.0),
        userNumMovies(USER_NUM, 0),
        Pu(USER_NUM, vector<double>(numFactor, 0.0)),
        Qi(MOVIE_NUM, vector<double>(numFactor, 0.0)),
        y(MOVIE_NUM, vector<double>(numFactor, 0.0)),
        sumYj(USER_NUM, vector<double>(numFactor, 0.0)) {

    ifstream fTrain(trainFile, ios::in);
    int userId, movieId, date, rating;
    int preUserId = 0; // previous userId
    int numMoviesCnt = 0;
    while (fTrain >> userId >> movieId >> date >> rating) {
        if (userId != preUserId) {
            userNumMovies.at(preUserId) = numMoviesCnt; // save the number of movies for each user
            numMoviesCnt = 0;
        }
        ++numMoviesCnt;

        Data dataPoint;
        dataPoint.userId = userId;
        dataPoint.movieId = movieId;
        dataPoint.date = date;
        dataPoint.rating = rating;
        trainData.push_back(dataPoint); // trainData, save each data point
        preUserId = userId;
    }
    userNumMovies.at(preUserId) = numMoviesCnt;

    if (!fTrain.eof()) throw runtime_error("Invalid data from trainFile!");
    fTrain.close();

    for (int i = 0; i < MOVIE_NUM; ++i) {
        for (int j = 0; j < numFactor; ++j) {
            double U1 = rand() / (RAND_MAX + 1.0);
            double V1 = rand() / (RAND_MAX + 1.0);
            Qi[i][j] = 0.1 * sqrt(-2.0 * log(U1)) *
                       sin(2.0 * PI * V1); // initialization to normal distribution with mu=0, sigma=0.1
            y[i][j] = 0.0;
        }
    }
    for (int i = 0; i < USER_NUM; i++) {
        for (int j = 0; j < numFactor; j++) {
            double U2 = rand() / (RAND_MAX + 1.0);
            double V2 = rand() / (RAND_MAX + 1.0);
//            sumYj[i][j] = 0.1 * sqrt(-2.0 * log(U2)) *
//                          cos(2.0 * PI * V2); // initialization to normal distribution with mu=0, sigma=0.1
            sumYj[i][j] = 0.0;
            Pu[i][j] = 0.1 * sqrt(-2.0 * log(U2)) *
                       sin(2.0 * PI * V2); // initialization to normal distribution with mu=0, sigma=0.1
        }
    }
}
