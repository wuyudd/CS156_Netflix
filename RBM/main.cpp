#define debug 0
#include "SparseMatrix.h"
#include "RBM.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <ctime>
#include "math.h"

using namespace std;
void query_parser(std::string file_name,
                  unordered_map<long, vector<int> >& query){
    std::ifstream file;
    file.open(file_name);
    if (not file.is_open()){
        std::cout << "Unable to open the data file,check the directory for file" << std::endl;
        exit(1);
    }
    std::string line;
    while (!file.eof()){
        getline(file, line);
        std::istringstream lineis(line);
        long user;
        int movie, date, rating;
        lineis >> user >> movie >> date >> rating;
        user -= 1;
        movie -= 1;
        if (query.find(user) == query.end()){
            std::vector<int> temp;
            query[user] = temp;
        }
        query[user].push_back(movie);
    }
}

void known_parser(std::string file_name, 
                     unordered_map<long, vector<int> >& query,
                     unordered_map<long, vector<double> >& rating){
    std::ifstream file;
    file.open(file_name);
    if (not file.is_open()){
        std::cout << "Unable to open the data file,check the directory for file" << std::endl;
        exit(1);
    }
    std::string line;
    while (!file.eof()){
        getline(file, line);
        std::istringstream lineis(line);
        long user;
        int movie, date;
        double rate;
        lineis >> user >> movie >> date >> rate;
        user -= 1;
        movie -= 1;
        if (query.find(user) == query.end()){
            std::vector<int> temp;
            query[user] = temp;
        }
        if (rating.find(user) == rating.end()){
            std::vector<double> temp;
            rating[user] = temp;
        }
        query[user].push_back(movie);
        rating[user].push_back(rate);
    }
}


int main() {
    //parameters
    long num_user = 458293;
    int num_movie = 17770;
    int num_rating = 5;

    //hyperparameters
    double learning_rate = 2e-5; 
    double weight_decay = 0.999;
    int epochs = 150;
    int k = 1;
    int num_hidden = 100;

    //flag
    bool save = true;
    bool check_validation = true;
    bool check_hidden = false;
    bool check_probe = true;

    //saving freqeucy
    int record_frequency = 3;

    //data path
    std::string base_data_path = "../um/valid.txt";
    std::string valid_data_path = "../um/valid.txt";
    std::string hidden_data_path = "../um/hidden.txt";
    std::string probe_data_path = "../um/probe.txt";
    std::string qual_data_path = "../um/qual.txt";

    //initialize the data as SparseMatrix
    SparseMatrix data = SparseMatrix(num_user, num_movie, num_rating, base_data_path);
    cout << "Finish initializing sparseMatrix" << endl;

    //initialize RBM network
    unordered_map<int, vector<long> > movies_info = data.movies_info;
    unordered_map<int, long> movies_num_ratings = data.movies_num_ratings;
    RBM rbm(num_user, num_movie, num_hidden, num_rating, movies_info, movies_num_ratings);
    cout << "Finish initializing RBM" << endl;

    //loading data
    std::cout << "Start Loading Valid Data" << std::endl;
    unordered_map<long, vector<int> > valid_query;
    unordered_map<long, vector<double> > valid_rating;
    //pass by reference to change valid_query and valid_rating
    known_parser(valid_data_path ,valid_query, valid_rating);
    std::cout << "Finish Loading Valid Data" << std::endl;

    std::cout << "Start Loading Hidden Data" << std::endl;
    unordered_map<long, vector<int> > hidden_query;
    unordered_map<long, vector<double> > hidden_rating;
    //pass by reference to change valid_query and valid_rating
    known_parser(hidden_data_path ,hidden_query, hidden_rating);
    std::cout << "Finish Loading Hidden Data" << std::endl;

    std::cout << "Start Loading Probe Data" << std::endl;
    unordered_map<long, vector<int> > probe_query;
    unordered_map<long, vector<double> > probe_rating;
    //pass by reference to change valid_query and valid_rating
    known_parser(probe_data_path ,probe_query, probe_rating);
    std::cout << "Finish Loading Probe Data" << std::endl;

    std::cout << "Start Loading Qual Data" << std::endl;
    unordered_map<long, vector<int> > qual_query;
    query_parser(qual_data_path, qual_query);
    std::cout << "Finish Loading Qual Data" << std::endl;

    //training
    for(int epoch = 0; epoch < epochs; epoch++){
    	cout << "Epoch: " << epoch << endl;
        //initialize timer
        clock_t start;
        double duration;
        start = clock();
        for(auto u : data.matrix){
            //cout << "User: " << u.first << "\r" << flush;
        	unordered_map<int, vector<double> > user_data = data.getWatchedMovie(u.first);
            rbm.contrastive_divergence(user_data, learning_rate, k);  
            if (debug) cout << "2" << endl;
        }
        cout << "Finish epoch: "<< epoch << endl;
        duration = (clock() - start)/ (double) CLOCKS_PER_SEC / (double) 60;
        cout << "Current epoch takes " << duration << " mins" << endl; 

        //calculate the validation loss
        if (check_validation){
            long num_m = 0;
            cout << "Start calculating validation loss ..." << endl;
            double valid_loss = 0.0;
            for(auto u:valid_query){
                //pass in the unordered_map<int, vector<double> >, vector<int>
                unordered_map<int, vector<double> > user_data = data.getWatchedMovie(u.first);
                vector<double> rating = rbm.reconstruct_multiple(user_data, u.second);
                for(int i = 0; i < rating.size(); i ++){
                    valid_loss += pow((rating[i] - valid_rating[u.first][i]),2);
                    num_m ++;
                }
            }
        printf("Validation loss is: %.20f \n",sqrt(valid_loss/num_m));
        }

        if (check_hidden){
            long num_m = 0;
            cout << "Start calculating hidden loss ..." << endl;
            double hidden_loss = 0.0;
            for(auto u:hidden_query){
                //pass in the unordered_map<int, vector<double> >, vector<int>
                unordered_map<int, vector<double> > user_data = data.getWatchedMovie(u.first);
                vector<double> rating = rbm.reconstruct_multiple(user_data, u.second);
                for(int i = 0; i < rating.size(); i ++){
                    hidden_loss += pow((rating[i] - hidden_rating[u.first][i]),2);
                    num_m ++;
                }
            }
            printf("Hidden loss is: %.20f \n",sqrt(hidden_loss/num_m));
        }

        if (check_probe){
            long num_m = 0;
            cout << "Start calculating probe loss ..." << endl;
            double probe_loss = 0.0;
            for(auto u:probe_query){
                //pass in the unordered_map<int, vector<double> >, vector<int>
                unordered_map<int, vector<double> > user_data = data.getWatchedMovie(u.first);
                vector<double> rating = rbm.reconstruct_multiple(user_data, u.second);
                for(int i = 0; i < rating.size(); i ++){
                    probe_loss += pow((rating[i] - probe_rating[u.first][i]),2);
                    num_m ++;
                }
            }
            printf("Probe loss is: %.20f \n",sqrt(probe_loss/num_m));
        }

        //every record_frequency epochs, write out the result for qual data.

        if (epoch % record_frequency == 0 && save){
            cout << "Calculating and saving the result for probe data ..." << endl;
            //make a new text file
            ofstream probefile;
            probefile.open ("../RBM_probe_output/epoch_"+ to_string(epoch) + "_probe.txt");
            //visit all user and write rating to files
            for(long user_idx = 0; user_idx < num_user; user_idx++){
                if (probe_query.find(user_idx) != probe_query.end()){
                    unordered_map<int, vector<double> > user_data = data.getWatchedMovie(user_idx);
                    vector<double> rating = rbm.reconstruct_multiple(user_data, probe_query[user_idx]);
                    for(auto r:rating){
                        probefile << to_string(r) + '\n';
                    }
                }
            }
            probefile.close();

            cout << "Calculating and saving the result for qual data ..." << endl;
            //make a new text file
            ofstream qualfile;
            qualfile.open ("../RBM_qual_output/epoch_"+ to_string(epoch) + "_qual.txt");
            //visit all user and write rating to files
            for(long user_idx = 0; user_idx < num_user; user_idx++){
                if (qual_query.find(user_idx) != qual_query.end()){
                    unordered_map<int, vector<double> > user_data = data.getWatchedMovie(user_idx);
                    vector<double> rating = rbm.reconstruct_multiple(user_data, qual_query[user_idx]);
                    for(auto r:rating){
                        qualfile << to_string(r) + '\n';
                    }
                }
            }
            qualfile.close();

            //save the rbm model
            rbm.save_model("../model/epoch_" + to_string(epoch) + "_W.txt",
                           "../model/epoch_" + to_string(epoch) + "_vb.txt",
                           "../model/epoch_" + to_string(epoch) + "_hb.txt");
        }

        if(epoch == 18){
            k = 3;
        }
        if(epoch == 27){
            k = 5;
        }
        if(epoch == 42){
            k = 9;
        }

        //update learning rate according to weight decay
        learning_rate = learning_rate * weight_decay; 
    }
    return 0;
}
