
#include "snake.h"


void Nash::initialize(int N_, int M_){
    N = N_;
    M = M_;
    A = new double*[N];
    for(int i=0; i<N; i++){
        A[i] = new double[M];
    }
    p1 = new double[N];
    for(int i=0; i<N; i++){
        p1[i] = 1.0 / N;
    }
    p2 = new double[M];
    for(int i=0; i<M; i++){
        p2[i] = 1.0 / M;
    }
    grad1 = new double[N];
    grad2 = new double[M];
    next1 = new double[N];
    next2 = new double[M];
}

void Nash::compute_gradients(double* p1, double* p2){
    for(int i=0; i<N; i++){
        double sum = 0;
        for(int j=0; j<M; j++){
            sum += A[i][j] * p2[j];
        }
        grad1[i] = sum;
    }
    for(int j=0; j<M; j++){
        double sum = 0;
        for(int i=0; i<N; i++){
            sum += A[i][j] * p1[i];
        }
        grad2[j] = sum;
    }
}

void Nash::compute_step(double* p1, double* p2, double step_size){
    exp_step(p1, grad1, next1, -step_size, N);
    exp_step(p2, grad2, next2, step_size, M);
}

void Nash::exp_step(double* policy, double* grad, double* next_policy, double step_size, int size){
    double sum = 0;
    for(int i=0; i<size; i++){
        next_policy[i] = policy[i] * exp(grad[i] * step_size);
        sum += next_policy[i];
    }
    for(int i=0; i<size; i++){
        next_policy[i] /= sum;
    }
}

void Nash::check_policy(double* policy, int size){
    double sum = 0;
    for(int i=0; i<size; i++){
        sum += policy[i];
    }
    assert(abs(sum - 1) < 1e-8);
}

void Nash::find_equilibrium(int iter, double threshold){
    for(int i=0; i<iter; i++){
        compute_gradients(p1, p2);
        compute_step(p1, p2, sqrt(rate)*alpha);
        compute_gradients(next1, next2);
        compute_step(p1, p2, rate);
        memcpy(p1, next1, N * sizeof(double));
        memcpy(p2, next2, M * sizeof(double));
        check_policy(p1, N);
        check_policy(p2, M);
        if(i % 100 == 0){
            if(exploitabilty() < threshold){
                break;
            }
        }
    }
}

double Nash::exploitabilty(){
    compute_gradients(p1, p2);
    double sum1 = 0;
    double max1 = -inf;
    for(int i=0; i<N; i++){
        sum1 += -grad1[i] * p1[i];
        max1 = max(-grad1[i], max1);
    }
    assert(max1 >= sum1 - 1e-08);
    double sum2 = 0;
    double max2 = -inf;
    for(int i=0; i<M; i++){
        sum2 += grad2[i] * p2[i];
        max2 = max(grad2[i], max2);
    }
    assert(max2 >= sum2 - 1e-08);
    return (max1 - sum1) + (max2 - sum2);
}

void Nash::evaluate(int size, int iter, int num_trial){
    double sum = 0;
    for(int t=0; t<num_trial; t++){
        int N_ = rand() % size + 1;
        int M_ = rand() % size + 1;
        initialize(N_, M_);
        for(int i=0; i<N; i++){
            for(int j=0; j<M; j++){
                A[i][j] = (double) rand() / RAND_MAX;
            }
        }
        find_equilibrium(iter, 0);
        sum += exploitabilty();
    }
    cout<<"AVERAGE EXPLOITABILITY: "<<(sum / num_trial)<<'\n';
}