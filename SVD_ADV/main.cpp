#include <iostream>
#include <sstream>
#include "SVD.h"

using namespace std;

int main() {
    // the parameters are set according to Koren's paper
    double learningrate = 0.005; // learning rate
    double regularization = 0.02; // regularization
    int numFactor = 20; // number of factors K
    int maxIter = 100; // number of iterations
    string trainFile = "../train.txt"; // train on index != 5 data
    string testFile = "../test.txt"; // test on qual data
    string probeFile = "../probe_with_rating.txt"; // get probe data error

    ostringstream oss;
    oss << "../predictions/svdadv_K-" << numFactor << "_Iter-" << maxIter << ".txt"; // output file path
    string outFile = oss.str();

    SVD svdadv(trainFile, testFile, probeFile, outFile, learningrate, regularization, numFactor, maxIter);
    double errorIn = svdadv.Train(); // in-sample error
    cout << "Final in-sample RMSE = " << errorIn << endl;
    return 0;
}