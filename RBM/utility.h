#pragma once
#include <iostream>
#include <math.h>
// using namespace std;

namespace utils{
    
    double sigmoid(double x){
        return 1.0/(1.0+exp(-x));
    }
    
    int binomial(double p){
        if(p<0 || p>1) return 0;
        
        int b = 0;
        double prob = (double) rand() / (double) RAND_MAX;

        if(prob < p) b+=1;

        return b;
    }
}
    

