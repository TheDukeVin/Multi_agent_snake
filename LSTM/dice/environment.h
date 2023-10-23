
#include "lstm.h"

#ifndef environment_h
#define environment_h

#define TIME_HORIZON 100
#define NUM_ACTIONS 2
#define DISCOUNT_FACTOR 1

/*
Dice Game
    A dice is rolled.
    Agent decides at each time step whether to reroll dice.
    Rerolled dice is added to added to sum
    If agent decides to end, total sum is given as reward
    If sum equals multiple of 10, game ends and 0 reward given.
*/

const int diceSize = 6;
const int bombMult = 10;

const int maxSum = 60;

#define inputSize maxSum

// double initParam = 0.1;
// double learnRate = 0.005;
// double momentum = 0;
// int batchSize = 30;
// double meanUpdate = 0.0001;

class Environment{
private:
    int sum;
    int diceVal;

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