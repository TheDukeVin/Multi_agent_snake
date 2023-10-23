
#include "environment.h"

Environment::Environment(){
    time = 0;
    endState = false;
    prevNum = -1;
    num = rand() % inputSize;
}

string Environment::toString(){
    return "Time: " + to_string(time) + " prev: " + to_string(prevNum);
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int i=0; i<size; i++){
        actions.push_back(i);
    }
    return actions;
}

double Environment::makeAction(int* actions){
    double reward = 0;
    if(prevNum != -1){
        double sum = 0;
        for(int i=0; i<size; i++){
            sum += actions[i] * (1 << i);
        }
        if(sum == prevNum){
            reward = 1;
        }
    }
    prevNum = num;
    num = rand() % inputSize;
    time ++;
    if(time == TIME_HORIZON){
        endState = true;
    }
    return reward;
}

void Environment::inputObservations(Data* input){
    for(int i=0; i<inputSize; i++){
        input->data[i] = 0;
    }
    input->data[num] = 1;
}