
#include "environment.h"

Environment::Environment(){
    time = 0;
    endState = false;
}

string Environment::toString(){
}

vector<int> Environment::validActions(){
}

double Environment::makeAction(int action){
    double reward = 0;
    time ++;
    if(time == TIME_HORIZON){
        endState = true;
    }
    return reward;
}

void Environment::inputObservations(Data* input){
}