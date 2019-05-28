#ifndef SVD_H_INCLUDED
#define SVD_H_INCLUDED

#include <vector>
#include <string>

using namespace std;

typedef struct Data
{
    int userId = 0;
    int movieId = 0;
    int date = 0;
    int rating = 0;
} Data;

class SVD {
public:
    SVD(const string &trainFile, const string &testFile, const string &probeFile, const string &outFile, double learningrate, double regularization,
        int numFactor, int maxIter);
    virtual ~SVD() {}
    double Train();

private:
    vector<double> Bi;
    vector<double> Bu;
    vector<vector<double> > Qi;
    vector<vector<double> > Pu;
    string trainFile;
    string testFile;
    string outFile;
    string probeFile;
    double learningrate;
    double regularization;
    int numFactor;
    int maxIter;
    vector<Data> trainData;

    void TrainInternal();
    void SaveTemp(int);
    void SaveQual();
    double ErrorIn();
    double ErrorOut();
    double PredictRating(int, int);
};

#endif // SVD_H_INCLUDED
