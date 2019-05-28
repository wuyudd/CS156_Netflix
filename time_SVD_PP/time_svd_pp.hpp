//
//  time_svd_pp.hpp
//  time_svd_pp
//
//  Created by Diyi Liu.
//  Copyright © 2019 Diyi Liu. All rights reserved.
//

#ifndef time_svd_pp_hpp
#define time_svd_pp_hpp

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>

using namespace std;

typedef struct Data
{
    int user = 0;
    int movie = 0;
    int date = 0;
    int rating = 0;
} Data;

class TimeSVDpp{
public:
    TimeSVDpp(const string&, const string&, const string&, const string&, const string&, const string&, double, double, double, double, double, double, double, int, int, int);
    virtual ~TimeSVDpp() {}
    void Learning();
    
protected:
    string train_file;
    string validation_file;
    string qual_file;
    string qual_out_file;
    string probe_file;
    string probe_out_file;
    vector<int> user_num_movies;
    vector<Data> train_data;
    
    double lr_alpha;
    double lr_init;
    double lr_decay;
    double la_alpha;
    double la_PQY;
    double la_bias;
    double threshold;
    
    int bin_num;
    int factor;
    int max_iter;
    
    vector<double> Bi;
    vector<double> Bu;
    vector<vector<double> > Qi;
    vector<vector<double> > Pu;
    vector<vector<double> > y;
    vector<vector<double> > Bi_bin;
    vector<double> alpha_u;
    vector<map<int,double> > Bu_t;
    vector<map<int,double> > Dev;
    vector<vector<double> > sum_Pu;
    vector<double> mu_Tu;
    
    double DevUt(int, int);
    int DateBin(int);
    double InError();
    double Predict(int, int, int);
    void Train();
    void SaveTemp(size_t);
    void PredictProbe();
};

#endif /* time_svd_pp_hpp */
