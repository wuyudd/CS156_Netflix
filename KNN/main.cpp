#include <iostream>
#include <sstream>
#include "knn.h"
using namespace std;

int main() {
    
    int K = 16; // k nearest neighbors
    string train_file = "../train.txt";
    string test_file = "../test.txt";

    ostringstream oss;
    oss << "../predictions/knn_without_default_K_" << K << "_qual.txt"; // output file path
    string out_file = oss.str();
    
    KNN knn(train_file, test_file, out_file, K);
    knn.Query();
    return 0;
}

