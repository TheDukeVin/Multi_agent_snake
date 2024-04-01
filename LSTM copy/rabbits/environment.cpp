
#include "environment.h"

Environment::Environment(){
    time = 0;
    endState = false;

    rabbits[0] = Pos(boardSize/2, boardSize/2);
    rabbits[1] = Pos(boardSize/2-1, boardSize/2);
    rabbits[2] = Pos(boardSize/2, boardSize/2-1);

    placeCarrot();
}

void Environment::placeCarrot(){
    carrotType = rand() % numRabbits;
    do {
        carrot = Pos(rand() % boardSize, rand() % boardSize);
    } while(rabbits[carrotType] == carrot);
}

string Environment::toString(){
    string s = "";
    for(int i=0; i<boardSize; i++){
        for(int j=0; j<boardSize; j++){
            Pos p(i, j);
            bool isRabbit = false;
            for(int k=0; k<numRabbits; k++){
                if(p == rabbits[k]){
                    s += to_string(k);
                    isRabbit = true;
                }
            }
            if(!isRabbit){
                s += '.';
            }
            if(p == carrot){
                s += "C ";
            }
            else{
                s += "  ";
            }
        }
        s += '\n';
    }
    s += "Carrot Type: " + to_string(carrotType) + '\n';
    return s;
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int r=0; r<numRabbits; r++){
        for(int d=0; d<8; d++){
            Pos curr = rabbits[r];
            bool jump = false;
            while(true){
                curr = curr.shift(d);
                if(!curr.inBounds()) break;
                if(jump){
                    actions.push_back(r*boardArea + curr.toIndex());
                }
                for(int i=0; i<numRabbits; i++){
                    if(curr == rabbits[i]){
                        jump = true;
                    }
                }
            }
        }
    }
    return actions;
}

double Environment::makeAction(int action){
    double reward = 0;
    int rabbitIndex = action / boardArea;
    Pos p(action % boardArea);
    rabbits[rabbitIndex] = p;
    if(p == carrot && rabbitIndex == carrotType){
        reward = 1;
        placeCarrot();
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
    for(int i=0; i<numRabbits; i++){
        input->data[i*boardArea + rabbits[i].toIndex()] = 1;
    }
    input->data[(3 + carrotType)*boardArea + carrot.toIndex()] = 1;
}