
#include "snake.h"

Kmeans::Kmeans(double* data_, int dim, int s){
    dimension = dim;
    size = s;
    data = new double*[size];
    for(int i=0; i<size; i++){
        data[i] = new double[dimension];
        for(int j=0; j<dimension; j++){
            data[i][j] = data_[i*dimension+j];
        }
    }
}

void Kmeans::cluster(int numClusters_, int maxIter, double tolerance){
    numClusters = numClusters_;
    centers = new double*[numClusters];
    for(int i=0; i<numClusters; i++){
        centers[i] = new double[dimension];
        // initialize to a random data point.
        int index = rand() % size;
        for(int j=0; j<dimension; j++){
            centers[i][j] = data[index][j] + (2*(rand()/RAND_MAX)-1) * 0.001;
        }
    }
    for(int i=0; i<maxIter; i++){
        cout<<"Iteration " << i << " of " << maxIter << '\n';
        if(update(tolerance)){
            break;
        }
    }
}

bool Kmeans::update(double tolerance){
    vector<int> clusters[numClusters];
    for(int i=0; i<size; i++){
        // find closest center
        int closest = -1;
        double minNorm = 1000*dimension;
        for(int j=0; j<numClusters; j++){
            double norm = 0;
            for(int k=0; k<dimension; k++){
                norm += squ(data[i][k] - centers[j][k]);
            }
            if(norm < minNorm){
                minNorm = norm;
                closest = j;
            }
        }
        assert(closest != -1);
        clusters[closest].push_back(i);
    }
    // update clusters to means
    bool converged = true;
    double maxDiff = 0;
    for(int i=0; i<numClusters; i++){
        double newCenter[dimension];
        for(int j=0; j<dimension; j++){
            newCenter[j] = 0;
        }
        for(auto d : clusters[i]){
            for(int j=0; j<dimension; j++){
                newCenter[j] += data[d][j];
            }
        }
        for(int j=0; j<dimension; j++){
            double diff = abs(centers[i][j] - newCenter[j] / clusters[i].size());
            maxDiff = max(maxDiff, diff);
            centers[i][j] = newCenter[j] / clusters[i].size();
        }
    }
    cout<<"Max Diff: "<<maxDiff<<'\n';
    return maxDiff < tolerance;
}