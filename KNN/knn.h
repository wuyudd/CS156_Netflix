#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class KNN {
    public :
    KNN(const string &train_file, const string &test_file, const string &out_file, int K);
    virtual ~KNN() {};
    void Query();
    
private:
    vector<vector<pair<int, int> > > user_movie;
    vector<vector<pair<int, int> > > movie_user;
    vector<vector<pair<int, float> > > pearson_matrix;
    vector<float> movie_mean;
    string train_file;
    string test_file;
    string out_file;
    int K;
    vector<double> avg_user_raings;
    vector<double> avg_movie_ratings;
    struct PearsonIntermediate {
        int i; // sum of rates all common users give to movie1
        int j; // sum of rates all common users give to movie2
        int ii; // sum of rates^2 all common users give to movie1
        int jj; // sum of rates^2 all common users give to movie12
        int ij; // sum of rates all common users give to movie1 * rates all common users give to movie2
        int cnt; // number of common users rating movies1 and movies2
    };
    void PearsonCoeff();
    void SaveQual();
    double PredictRating(int, int);
};

#endif //KNN_H


