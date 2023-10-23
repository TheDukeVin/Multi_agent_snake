
#include "environment.h"

Environment::Environment(){
    time = 0;
    endState = false;
}

string Environment::toString(){
}

vector<int> Environment::validActions(){
    vector<int> actions;
    return actions;
}

double Environment::makeAction(int* actions){
    double reward = 0;
    time ++;
    if(time == TIME_HORIZON){
        endState = true;
    }
    return reward;
}

void Environment::inputObservations(Data* input){
}