
#include "PG.h"

int sampleDist(double* dist, int N){
    double sum = 0;
    for(int i=0; i<N; i++){
        if(dist[i] >= 0) sum += dist[i];
    }
    if(abs(sum - 1) >= 1e-07){
        ofstream errOut("err.out");
        errOut<<"Invalid distribution: \n";
        for(int i=0; i<N; i++){
            errOut<<dist[i]<<' ';
        }
        errOut<<'\n';
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
