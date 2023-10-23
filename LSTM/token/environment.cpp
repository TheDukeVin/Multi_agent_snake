
#include "environment.h"

Environment::Environment(){
    time = 0;
    endState = false;

    runner = Pos(boardSize/2, boardSize/2);
    placeToken();
}

string Environment::toString(){
    return to_string(runner.x) + " " + to_string(runner.y) + " " + to_string(token.x) + " " + to_string(token.y);
}

void Environment::placeToken(){
    do {
        token = Pos(rand() % boardSize, rand() % boardSize);
    } while(token == runner);
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int i=0; i<4; i++){
        Pos next = runner.shift(i);
        if(next.inBounds()){
            actions.push_back(i);
        }
    }
    return actions;
}

double Environment::makeAction(int action){
    runner = runner.shift(action);
    if(runner == token){
        placeToken();
        return 1;
    }
    return 0;
}

void Environment::inputObservations(Data* input){
    for(int i=0; i<inputSize; i++){
        input->data[i] = 0;
    }
    input->data[runner.x] = 1;
    input->data[runner.y + boardSize] = 1;
    if(abs(runner.x - token.x) <= proximity && abs(runner.y - token.y) <= proximity){
        input->data[token.x + boardSize*2] = 1;
        input->data[token.y + boardSize*3] = 1;
    }
}