//
//  main.cpp
//  time_svd_pp
//
//  Created by Diyi Liu.
//  Copyright © 2019 Diyi Liu. All rights reserved.
//
#include "time_svd_pp.hpp"
#include <iostream>
#include <sstream>


using namespace std;

int main() {
    
    // Learning rate
    double lr_alpha = 0.0001;
    double lr_init = 0.007;
    double lr_decay = 0.9;
    // Regularization
    double la_alpha = 0.01;
    double la_PQY = 0.015;
    double la_Bias = 0.01;
    double threshold = 0.00005;
    
    int bin_num = 30;
    int factor = 80;
    int max_iter = 50;
    
    string train_file = "../train.txt";
    string validation_file = "../valid.txt";
    
    string qual_file = "../test.txt";
    ostringstream oss1;
    oss1 << "../predictions/time_svd_pp_qual_k" << factor << "_Iter-" << max_iter << ".txt";
    string qual_out_file = oss1.str();


    string probe_file = "../probe.txt";
    ostringstream oss2;
    oss2 << "../predictions/time_svd_pp_probe_k" << factor << "_Iter-" << max_iter << ".txt";
    string probe_out_file = oss2.str();
    
    TimeSVDpp timesvdpp(train_file, validation_file, qual_file, qual_out_file, probe_file, probe_out_file, lr_init, lr_alpha, lr_decay, la_Bias, la_PQY, la_alpha, threshold, bin_num, factor, max_iter);
    timesvdpp.Learning();
    
    return 0;
}
