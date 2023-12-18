
#include "lstm.h"

using namespace LSTM;

Params::Params(int size_){
    size = size_;
    params = new double[size];
    gradient = new double[size];
    for(int i=0; i<size; i++){
        params[i] = gradient[i] = 0;
    }
}

void Params::randomize(double scale){
    for(int i=0; i<size; i++){
        params[i] = (2 * (double) rand() / RAND_MAX - 1) * scale;
    }
}

void Params::copy(Params params_){
    for(int i=0; i<size; i++){
        params[i] = params_.params[i];
        gradient[i] = params_.gradient[i];
    }
}

void Params::accumulateGradient(Params params_){
    for(int i=0; i<size; i++){
        gradient[i] += params_.gradient[i];
    }
}

void Params::update(double scale, double momentum, double regRate){
    for(int i=0; i<size; i++){
        params[i] -= gradient[i] * scale;
        params[i] *= 1-regRate;
        gradient[i] *= momentum;
        assert(abs(params[i]) < 1000);
    }
}

void Params::resetGradient(){
    for(int i=0; i<size; i++){
        gradient[i] = 0;
    }
}