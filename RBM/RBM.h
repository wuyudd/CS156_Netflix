#include <unordered_map>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <random>
using namespace std;



class RBM{
    
public:
    long N;
    int num_vis;
    int num_hid;
    int num_rat;
    double ***W;
    double *h_b;
    double **v_b;
    RBM(long , int , int , int, unordered_map<int, vector<long> >&, unordered_map<int, long>&);
    void save_model(string, string, string);
    void load_model(string, string, string);
    ~RBM();
    void contrastive_divergence(unordered_map<int,vector<double> >&, double, int);
    double forward(unordered_map<int, vector<double> >& , int);
    double backward(double *, int, int);
    void sample_hgv(unordered_map<int, vector<double> >&, double *, double *);
    void sample_vgh(double *, unordered_map<int, vector<double> >&, unordered_map<int, vector<double> >&);
    void gibbs_sampling(double*, 
                        unordered_map<int, vector<double> > &,
                        unordered_map<int, vector<double> >&,
                        double*,
                        double* 
                        );
    double reconstruct_single(unordered_map<int,vector<double> >&, int);
    vector<double> reconstruct_multiple(unordered_map<int,vector<double> >&, vector<int>);
};