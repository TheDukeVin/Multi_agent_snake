
#include "lstm.h"

#ifndef environment_h
#define environment_h

#define TIME_HORIZON 100
#define NUM_ACTIONS 4
#define DISCOUNT_FACTOR 0.98

#define inputSize (boardSize * 4)

class Environment{
private:

public:
    int time;
    bool endState;

    Environment();
    string toString();

    vector<int> validActions();
    double makeAction(int action); // returns reward

    void inputObservations(Data* input);
};

#endif