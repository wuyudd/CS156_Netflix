// Created by Yu Wu

#include <iostream>
#include <sstream>
#include "SVD.h"

using namespace std;

int main() {
    // the parameters are set according to Koren's paper
    double lrInit = 0.007; // learning rate
    double lrDecay = 0.9; // learning rate decay
    double laBias = 0.005; // regularization lambda of Bu Bi
    double laFactor = 0.015; // regularization lambda of Pu Qi y
    int numFactor = 80; // number of factors K
    int maxIter = 100; // number of iterations
    string trainFile = "../train.txt"; // train on index != 5 data
    string testFile = "../test.txt"; // test on qual data
    string probeFile = "../probe_with_rating.txt"; // get probe data error
    ostringstream oss;
    oss << "../predictions/old-svdpp_K-" << numFactor << "_Iter-" << maxIter << ".txt"; // output file path
    string outFile = oss.str();
    SVD svdpp(trainFile, testFile, probeFile, outFile, lrInit, lrDecay, laBias, laFactor, numFactor, maxIter);
    double errorIn = svdpp.Train(); // in-sample error
    cout << "Final in-sample RMSE = " << errorIn << endl;
    return 0;
}