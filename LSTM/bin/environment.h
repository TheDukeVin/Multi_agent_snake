
#include "lstm.h"

#ifndef environment_h
#define environment_h

#define size 3

#define TIME_HORIZON 10
#define NUM_ACTIONS size
#define DISCOUNT_FACTOR 1

#define inputSize (1 << size)

/*
Train settings

double initParam = 0.1;
double learnRate = 0.01;
double momentum = 0.7;
int batchSize = 20;
double regRate = 0.00001;
double meanUpdate = 0.001;

PG(){
    Model m(inputSize);
    m.addLSTM(10);
    m.addOutput(NUM_ACTIONS);
    seq = ModelSeq(m, TIME_HORIZON, initParam);
}

Train for 300000 steps, nonparallel

*/

class Environment{
private:
    int prevNum;
    int num;

public:
    int time;
    bool endState;

    Environment();
    string toString();

    vector<int> validActions();
    double makeAction(int* actions); // returns reward

    void inputObservations(Data* input);
};

#endif