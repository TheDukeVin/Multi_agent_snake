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


double abs(double x){
    if(x < 0) return -x;
    return x;
}

int sampleDist(double* dist, int N){
    double sum = 0;
    for(int i=0; i<N; i++){
        if(dist[i] >= 0) sum += dist[i];
    }
    if(abs(sum - 1) > 1e-07){
        cout<<"Bruh\n";
        for(int i=0; i<N; i++){
            cout<<dist[i]<<' ';
        }
        cout<<'\n';
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
        if(randReal <= parsum){
            index = i;
            break;
        }
    }
    assert(index != -1);
    return index;
}