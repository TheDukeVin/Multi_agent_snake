//
//  common.cpp
//  common
//
//  Created by Kevin Du on 1/6/23.
//


#include "snake.h"

double squ(double x){
    return x * x;
}

int max(int x, int y){
    if(x < y) return y;
    return x;
}

double max(double x, double y){
    if(x < y) return y;
    return x;
}

double min(double x, double y){
    if(x < y) return x;
    return y;
}


// double abs(double x){
//    if(x < 0) return -x;
//    return x;
// }

void computeSoftmaxPolicy(double* logits, vector<int> validActions, double* policy){
    double maxLogit = -1e+10;
    for(auto a : validActions){
        if(logits[a] > maxLogit){
            maxLogit = logits[a];
        }
    }
    double sum = 0;
    for(auto a : validActions){
        sum += exp(logits[a] - maxLogit);
    }
    for(auto a : validActions){
        policy[a] = exp(logits[a] - maxLogit) / sum;
    }
}

int sampleDist(double* dist, int N){
    double sum = 0;
    for(int i=0; i<N; i++){
        if(dist[i] >= 0) sum += dist[i];
    }
    if(abs(sum - 1) > 1e-07){
        string s = "Invalid distribution\n";
        for(int i=0; i<N; i++){
            s += to_string(dist[i]) + ' ';
        }
        s += '\n';
        cout<<s;
    }
    assert(abs(sum - 1) < 1e-07);

    double parsum = 0;
    double randReal = (double)rand() / RAND_MAX;
    
    int index = -1;
    for(int i=0; i<N; i++){
        if(dist[i] < 0){
            continue;
        }
        parsum += dist[i];
        if(randReal < parsum + 1e-06){
            index = i;
            break;
        }
    }
    assert(index != -1);
    return index;
}
