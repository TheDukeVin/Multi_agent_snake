
#include "lstm.h"

using namespace LSTM;

Params::Params(int size_){
    size = size_;
    params = new double[size];
    gradient = new double[size];
    first_moment = new double[size];
    second_moment = new double[size];
    for(int i=0; i<size; i++){
        params[i] = gradient[i] = 0;
        first_moment[i] = second_moment[i] = 0;
    }
    numUpdates = 0;
}

void Params::randomize(double scale){
    for(int i=0; i<size; i++){
        params[i] = (2 * (double) rand() / RAND_MAX - 1) * scale;
    }
}

void Params::copy(Params* params_){
    // cout << "Copy params " << params << ' ' << gradient << '\n';
    for(int i=0; i<size; i++){
        params[i] = params_->params[i];
        gradient[i] = params_->gradient[i];
    }
    // cout << "Copied\n";
}

void Params::accumulateGradient(Params* params_){
    for(int i=0; i<size; i++){
        gradient[i] += params_->gradient[i];
    }
}

void Params::update(double scale, double momentum, double regRate){
    numUpdates ++;
    double weight_1 = (1 - pow(beta_1, numUpdates)); // / (1 - beta_1);
    double weight_2 = (1 - pow(beta_2, numUpdates)); // / (1 - beta_2);

    for(int i=0; i<size; i++){
        gradient[i] = gradient[i] + 2*regRate * params[i];
        first_moment[i] = beta_1 * first_moment[i] + (1-beta_1) * gradient[i];
        // first_moment[i] *= beta_1;
        // first_moment[i] += gradient[i];
        second_moment[i] = beta_2 * second_moment[i] + (1-beta_2) * pow(gradient[i], 2);
        // second_moment[i] *= beta_2;
        // second_moment[i] += pow(gradient[i], 2);
        gradient[i] = 0;

        params[i] -= scale * (first_moment[i] / weight_1) / (sqrt(second_moment[i] / weight_2) + epsilon);
        // params[i] *= 1-regRate;

        assert(abs(params[i]) < 1000);
    }
}

void Params::resetGradient(){
    for(int i=0; i<size; i++){
        gradient[i] = 0;
    }
}