#define debug 0
#include "RBM.h"
#include "utility.h"


RBM::RBM(long size, int num_v, int num_h, int num_r, 
        unordered_map<int, vector<long> >& movies_info, 
        unordered_map<int, long>& movies_num_ratings){
    N = size;
    num_vis = num_v;
    num_hid = num_h;
    num_rat = num_r;
    //weight matrix:
    //first dimension is number of movies,
    //second dimension is number of ratings,
    //third dimension is number of features.
    W = new double** [num_vis];
    for(int i=0; i<num_vis; i++){
        W[i] = new double* [num_rat];
        for(int k=0; k<num_rat;k++){
            W[i][k] = new double[num_hid];
            for (int j=0; j<num_hid; ++j)
                W[i][k][j] = 0;
        }
    }

    //gaussian random variable generator
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,0.01);
    //initialize the weight matrix with normal distribution, mean = 0, variance = 0.01   
    for(int i = 0; i < num_vis; i++){
        for(int k = 0; k < num_rat; k++){
            for(int j = 0; j < num_hid; j++){
                W[i][k][j] = distribution(generator);
            }
        }
    }

    
    h_b = new double[num_hid];
    for(int i = 0; i < num_hid; i++){
        //hidden biases are initialized as zero
        //initialize the hidden bias with normal distribution, mean = 0, variance = 0.01
        h_b[i] = distribution(generator);
    }

    //using movies_info, movies_num_ratings to initialize the visible biases
    v_b = new double*[num_vis];
    //DONE:log[p/(1-p)]
    for(int i = 0; i < num_vis; i++){
        v_b[i] = new double[num_rat];
        //other initialization
        for(int k=0; k < num_rat; k++){
            v_b[i][k] = 0.0;
            //need to have movie i in the data
            //if not all users give the same rating to movie i
            //there must be someone giving the kth rating for movie i
            if (movies_info.find(i) != movies_info.end() 
                && movies_num_ratings[i] != movies_info[i][k] 
                && movies_info[i][k] != 0){
                v_b[i][k] = log( double(movies_info[i][k])/double(movies_num_ratings[i] - movies_info[i][k]) );
            }
        }
    }
}

void RBM::save_model(string W_path, string vb_path, string hb_path){
    cout << "Saving the model ..." << endl;
    //save the model to txt file
    ofstream myfile_model_w;
    ofstream myfile_model_vb;
    ofstream myfile_model_hb;
    myfile_model_w.open (W_path);
    myfile_model_vb.open (vb_path);
    myfile_model_hb.open (hb_path);
    for(int i=0; i<num_vis; i++){
        for(int k=0; k<num_rat; k++){
            if (k == num_rat - 1) {
                myfile_model_vb << to_string(v_b[i][k]) + "\n";
            }
            else{
                myfile_model_vb << to_string(v_b[i][k]) + " ";
            }

            for(int j=0; j<num_hid; j++){
                myfile_model_w << to_string(W[i][k][j]) << endl;
            }
        }
    }
    myfile_model_w.close();
    myfile_model_vb.close();
    
    for(int j=0; j<num_hid;j++){
        myfile_model_hb << to_string(h_b[j]) << endl;
    }
    myfile_model_hb.close();
}

//load model from existing file 
void RBM::load_model(string W_path, string vb_path, string hb_path){
    ifstream in_W;
    ifstream in_vb;
    ifstream in_hb;

    in_W.open(W_path);
    if (!in_W) {
        cout << "Unable to open file W";
        exit(1); // terminate with error
    }

    in_vb.open(vb_path);
    if (!in_vb) {
        cout << "Unable to open file vb";
        exit(1); // terminate with error
    }

    in_hb.open(hb_path);
    if (!in_hb) {
        cout << "Unable to open file hb";
        exit(1); // terminate with error
    }
    double x,y,z;
    int i = 0;
    int k = 0;
    int j = 0;
    while(in_W >> x){
        W[i][k][j]= x;
        if(j == num_hid-1){
            j = 0;
            if(k == num_rat-1){
            k = 0;
            i++;
            }else{
                k++;
            }
        }else{
            j++;
        }

        if(i == num_vis)break;
    }

    i = 0;
    k = 0;
    j = 0;
    while(in_vb >> y){
        v_b[i][k] = y;
        if(k == num_rat-1){
            k = 0;
            i++;
        }else{
            k++;
        }
    }

    i = 0;
    k = 0;
    j = 0;
    while(in_hb >> z){
        h_b[j] = z;
        j++;
    }
    in_W.close();
    in_vb.close();
    in_hb.close();
}


RBM::~RBM() {
    for(int i=0; i<num_vis; i++){
        for(int k=0; k<num_rat; k++){
            delete[] W[i][k];
        }
        delete[] W[i];
    }
    if (debug) cout << "destructor 1" << endl;
    for(int i=0; i<num_vis; i++){
        delete[] v_b[i];
    }
    if (debug) cout << "destructor 2" << endl;
    delete[] W;
    if (debug) cout << "destructor 3" << endl;
    delete[] h_b;
    if (debug) cout << "destructor 4" << endl;
    delete[] v_b;
    if (debug) cout << "destructor 5" << endl;
}

void RBM::contrastive_divergence(unordered_map<int, vector<double> >& input, double learning_rate, int steps) {

    double *first_hid_probs = new double[num_hid];
    double *first_hid_sample = new double[num_hid];
    double *new_hid_probs = new double[num_hid];
    double *new_hid_sample = new double[num_hid];
    for (int i=0; i<num_hid; ++i) {
        first_hid_probs[i] = 0;
        first_hid_sample[i] = 0;
        new_hid_probs[i] = 0;
        new_hid_sample[i] = 0;
    }
    
    unordered_map<int, vector<double> > new_vis_probs;
    //initialize new_vis_prob
    for(auto x: input){
        vector<double> temp(5, 0.0);
        new_vis_probs[x.first] = temp;
    }

    unordered_map<int, vector<double> > new_vis_samples;
    //initialize new_vis_prob
    for(auto x: input){
        vector<double> temp(5, 0.0);
        new_vis_samples[x.first] = temp;
    }
    if (debug) cout << "rbm c_d 1" << endl;
    
    sample_hgv(input, first_hid_probs, first_hid_sample);
    if (debug) cout << "sample_hgv" << endl;

    for(int s = 0; s < steps; s++){
        if(s == 0){
            gibbs_sampling(first_hid_sample, 
                           new_vis_probs, 
                           new_vis_samples,
                           new_hid_probs, 
                           new_hid_sample);
            if (debug) cout << "rbm c_d for s = 0" << endl;
        }else{
            gibbs_sampling(new_hid_sample, 
                           new_vis_probs, 
                           new_vis_samples,
                           new_hid_probs, 
                           new_hid_sample);
            if (debug) cout << "rbm c_d  for s > 0" << endl;
        }
    }
    if (debug) cout << "rbm c_d 2" << endl;
    for(auto x:input){
        for(int k = 0; k < num_rat; k++){
            for(int j = 0; j < num_hid; j ++){
                //update weight
                //use the difference between initial samples and final samples
                W[x.first][k][j] += learning_rate * (input[x.first][k] * first_hid_sample[j] - new_vis_samples[x.first][k] * new_hid_sample[j]);
            }
            //update visible unit biases
            //use the difference between initial samples and final samples
            v_b[x.first][k] += learning_rate * (input[x.first][k] - new_vis_samples[x.first][k]);
        }
        
    }
    if (debug) cout << "rbm c_d 3" << endl;
    //update hidden unit biases
    //use difference between initial samples and final samples
    for(int j = 0; j < num_hid; j ++){
        h_b[j] += learning_rate * (first_hid_sample[j] - new_hid_sample[j]);//first_hid_probs[j]-new_hid_probs[j]
    }

    if (debug) cout << "rbm c_d 4" << endl;
    delete[] first_hid_probs;
    delete[] first_hid_sample;
    delete[] new_hid_probs;
    delete[] new_hid_sample;

    
}

double RBM::forward(unordered_map<int, vector<double> > &vis, int j){

    //for jth hidden unit
    double sum_before_sigmoid=0.0;
    //visit all watched movie for this user
    for(auto movie: vis){
        //visit all ratings for current movies
        for(int k=0;k<num_rat;k++){
            sum_before_sigmoid += movie.second[k]*W[movie.first][k][j];
        }
    }
    //add bias for this hidden unit
    sum_before_sigmoid = sum_before_sigmoid + h_b[j];
    return utils::sigmoid(sum_before_sigmoid);
}
    
double RBM::backward(double* h, int i, int k){
    //for movie i, rating k

    //declare the numerator, denominator
    double num = 0.0;
    double den = 0.0;
    double den_sum = 0.0;

    for(int j = 0; j<num_hid; j++){
        num += h[j]*W[i][k][j];
    }
    //add bias for movie i rating k, and takes exponential.
    num = exp(v_b[i][k]+num);
    if (debug) cout << "backward exp" << num << endl;
    //visit all ratings
    for(int l=0; l<num_rat; l++){
        den_sum = 0;
        //visit all hidden units
        //collect all contribution from hidden units for kth rating of ith movie
        for(int j=0; j<num_hid; j++){
            den_sum += h[j]*W[i][l][j];
        }
        den += exp(v_b[i][l]+den_sum);
    }
    return (num/den);

}

//sample hidden unit given visible unit
void RBM::sample_hgv(unordered_map<int, vector<double> >& v0_sample, double *hidden_probs, double *hidden_samples){
    //v0_sample is the given visible unit

    for(int j=0; j<num_hid; j++){
        hidden_probs[j] = forward(v0_sample, j);
        // cout << "foward" << endl;
        hidden_samples[j] = utils::binomial(hidden_probs[j]);
        // cout << "binomial" << endl;
    }
}

//calculate visible unit probabilities given hidden samples
void RBM::sample_vgh(double* h0_sample, unordered_map<int, vector<double> >& probs, unordered_map<int, vector<double> >& vis_samples){
    //h0_sample is the very first input hidden space to this step of gibbs sampling
    for(auto movie: probs){
        for(int k=0; k<num_rat; k++){
            movie.second[k] = backward(h0_sample, movie.first, k);
            vis_samples[movie.first][k] = utils::binomial(movie.second[k]);
            if (debug) cout << "sample_vgh: " << k << endl;
        }
    }
}

//go backward once: visible given hidden
//go forward once: hidden given visible
void RBM::gibbs_sampling(double* hid_init, 
                         unordered_map<int, vector<double> > &vis_probs,
                         unordered_map<int, vector<double> > &vis_samples,
                         double* hid_probs,
                         double* hid_samples){
    sample_vgh(hid_init, vis_probs, vis_samples);
    if (debug) cout << "gibbs 1" << endl;
    sample_hgv(vis_samples, hid_probs, hid_samples);
}


double RBM::reconstruct_single(unordered_map<int, vector<double> >& input, int i){
    double *recon_h = new double[num_hid];
    
    //original forward
    for(int j = 0; j<num_hid; j++){
        recon_h[j] = forward(input, j);
    }
    //targeted backward for movie i
    vector<double> recon_probs(5, 0.0);
    double sum = 0.0;
    double prediction=0.0;

    //get probability for each rating of movie i
    for(int k = 0; k<num_rat; k++){
        recon_probs[k] = backward(recon_h, i, k);
        sum += recon_probs[k];
    }
    //normalize the probability
    for(int k = 0; k<num_rat; k++){
        recon_probs[k] /= sum;
    }
    //make prediction, calculate weighted total
    for(int k = 0; k<num_rat; k++){
        prediction += (k+1)*recon_probs[k];
    }
    delete[] recon_h;
    return prediction;
}

vector<double> RBM::reconstruct_multiple(unordered_map<int, vector<double> >& input, vector<int> mvs){
    double *recon_h = new double[num_hid];
    int size = mvs.size();
    vector<double> pred(size, 0.0);
    
    //original forward
    for(int j = 0; j<num_hid; j++){
        recon_h[j] = forward(input, j);
    }
    
    //backward for the set
    for(int i = 0; i< mvs.size(); i++){
        
        vector<double> recon_probs(5, 0.0);
        double sum = 0.0;
        double prediction=0.0;

        //get probability for each rating of movie i
        for(int k = 0; k<num_rat; k++){
            recon_probs[k] = backward(recon_h, mvs[i], k);
            sum += recon_probs[k];
        }
        //normalize the probability
        for(int k = 0; k<num_rat; k++){
            recon_probs[k] /= sum;
        }
        //make prediction, calculate weighted total
        for(int k = 0; k<num_rat; k++){
            prediction += (k+1)*recon_probs[k];
        }
        
        pred[i] = prediction;
    }
    delete[] recon_h;
    return pred;
    
}
