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
    ossInter << "../predictions/inter_svdadv_qual_K-" << numFactor << "_Iter-" << index + 1 << ".txt";
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

// Main train function: train, calculate in-sample error, and predict test data
double SVD::Train() {
    srand((unsigned)time(nullptr));
    double currRmse = 0.0;
    cout << "********** Starting training with M = " << USER_NUM << " users, N = " << MOVIE_NUM << " items. **********" << endl;
    cout << "********** Training with Factor K = " << numFactor << ", Max Iteration = " << maxIter << ". **********" << endl;
    for(int i = 0; i < maxIter; ++i) {
        cout << "Current Iteration = " << i + 1 << endl;
        TrainInternal();
        currRmse = ErrorIn();
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

    return sqrt(squErr / count);
}

// out-sample error
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
    cout << "       ErrorOut function: count = " << count << endl;
    return sqrt(squErr / count);
}

// predict the rating the userId gives to movieId
double SVD::PredictRating(int userId, int movieId){

    double tempRes = 0.0;
    for(int k = 0; k < numFactor; ++k) {
        tempRes += Pu[userId][k] * Qi[movieId][k];
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
	
    // go through every data point
    for (int i = 0; i < trainData.size(); ++i) {
        userId = trainData[i].userId;
        movieId = trainData[i].movieId;
        rating = trainData[i].rating;
		
		// calculate error for current data point and update Bu, Bi, Pu, Qi. 
        double diffRating = rating - PredictRating(userId, movieId);
        Bu[userId] += learningrate * (diffRating - regularization * Bu[userId]);
        Bi[movieId] += learningrate * (diffRating - regularization * Bi[movieId]);
        for (int k = 0; k < numFactor; ++k) {
            double puVal = Pu[userId][k];
            double qiVal = Qi[movieId][k];
            Pu[userId][k] += learningrate * (diffRating * qiVal - regularization * puVal);
            Qi[movieId][k] += learningrate * (diffRating * puVal - regularization * qiVal);
        }
    }
}


SVD::SVD(const string &trainFile, const string &testFile, const string &probeFile, const string &outFile, double learningrate, double regularization,
         int numFactor, int maxIter) :
        trainFile(trainFile),testFile(testFile), probeFile(probeFile), outFile(outFile),
        learningrate(learningrate), regularization(regularization),
        numFactor(numFactor), maxIter(maxIter),
        Bu(USER_NUM, 0.0),
        Bi(MOVIE_NUM, 0.0),
        Pu(USER_NUM, vector<double>(numFactor, 0.0)),
        Qi(MOVIE_NUM, vector<double>(numFactor, 0.0)) {
	// Load train data and save in struct
    ifstream fTrain(trainFile, ios::in);
    int userId, movieId, date, rating;
    while (fTrain >> userId >> movieId >> date >> rating) {
        Data dataPoint;
        dataPoint.userId = userId;
        dataPoint.movieId = movieId;
        dataPoint.date = date;
        dataPoint.rating = rating;
        trainData.push_back(dataPoint);
    }
	
    if (!fTrain.eof()) throw runtime_error("Invalid data from trainFile!");
    fTrain.close();
	
	// Initialization
    for (int i = 0; i < MOVIE_NUM; ++i) {
        for (int j = 0; j < numFactor; ++j) {
            double U1 = rand() / (RAND_MAX + 1.0);
            double V1 = rand() / (RAND_MAX + 1.0);
            Qi[i][j] = 0.1 * sqrt(-2.0 * log(U1)) *
                       sin(2.0 * PI * V1); // initialization to normal distribution with mu=0, sigma=0.1
        }
    }
    for (int i = 0; i < USER_NUM; i++) {
        for (int j = 0; j < numFactor; j++) {
            double U2 = rand() / (RAND_MAX + 1.0);
            double V2 = rand() / (RAND_MAX + 1.0);
            Pu[i][j] = 0.1 * sqrt(-2.0 * log(U2)) *
                       sin(2.0 * PI * V2); // initialization to normal distribution with mu=0, sigma=0.1
        }
    }
}
