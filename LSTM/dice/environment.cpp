
#include "environment.h"

Environment::Environment(){
    time = 0;
    endState = false;

    diceVal = rand() % diceSize;
    sum = diceVal + 1;
}

string Environment::toString(){
    return "Time: " + to_string(time) + ", Sum: " + to_string(sum) + ", Dice Val: " + to_string(diceVal);
}

vector<int> Environment::validActions(){
    return vector<int>{0, 1};
}

double Environment::makeAction(int action){
    double reward = 0;
    if(time == 0){
        reward += sum;
    }
    if(action == 0){
        endState = true;
    }
    else{
        diceVal = rand() % diceSize;
        sum += diceVal + 1;
        reward += diceVal + 1;
        if(sum % bombMult == 0){
            endState = true;
            reward -= sum;
        }
        if(sum >= maxSum){
            endState = true;
        }
    }
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
    input->data[sum] = 1;
    // input->data[0] = sum/10;
}