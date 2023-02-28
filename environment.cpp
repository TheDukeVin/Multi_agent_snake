//
//  environment.cpp
//  environment
//
//  Created by Kevin Du on 1/18/22.
//

#include "snake.h"

void Environment::setGridValue(Pos p, int val){
    grid[p.x][p.y] = val;
}

int Environment::getGridValue(Pos p){
    return grid[p.x][p.y];
}

void Environment::initialize(){
    timer = 0;
    actionType = 0;
    snakes[0].size = 2;
    snakes[0].head = Pos(boardx/2, 1);
    snakes[0].tail = Pos(boardx/2, 0);

    snakes[1].size = 2;
    snakes[1].head = Pos(boardx/2, boardy-2);
    snakes[1].tail = Pos(boardx/2, boardy-1);
    
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            grid[i][j] = -1;
        }
    }

    setGridValue(snakes[0].head, 4);
    setGridValue(snakes[0].tail, 0);
    setGridValue(snakes[1].head, 9);
    setGridValue(snakes[1].tail, 7);
    
    while(true){
        apple = Pos(rand() % boardx, rand() % boardy);
        if(getGridValue(apple) == -1){
            break;
        }
    }
}

bool Environment::isEndState(){
    if(timer == maxTime){
        return true;
    }
    for(int i=0; i<numAgents; i++){
        bool surrounded = true;
        for(int d=0; d<4; d++){
            if(validAgentAction(i, d)){
                surrounded = false;
                break;
            }
        }
        if(surrounded) return true;
    }
    return false;
}

bool Environment::validAgentAction(int agentID, int action){ // returns whether the action is valid.
    Pos nbr = snakes[agentID].head.shift(action);
    return nbr.inBounds() && (getGridValue(nbr) == -1);
}

// void Environment::setAgentAction(int agentID, int action){
//     assert(validAgentAction(agentID, action));
//     agentActions[agentID] = action;
// }

bool Environment::validChanceAction(int pos){
    int newApplex = pos / boardy;
    int newAppley = pos % boardy;
    return getGridValue(Pos(newApplex, newAppley)) == -1;
}

void Environment::makeAction(Action chosenAction){
    assert(actionType == chosenAction.actionType);
    if(actionType == 0){
        agentAction(chosenAction.agentActions);
    }
    else{
        chanceAction(chosenAction.chanceAction);
    }
    
    // Unfold path
    /*
    while(actionType == 0 && !isEndState()){
        int nextAction = -1;
        for(int i=0; i<numAgentActions; i++){
            if(validAction(i)){
                if(nextAction == -1){
                    nextAction = i;
                }
                else{
                    return;
                }
            }
        }
        agentAction(nextAction);
    }*/
}

// void Environment::setAction(Environment* currState, int actionIndex){
//     copyEnv(currState);
//     makeAction(actionIndex);
// }

void Environment::agentAction(int* agentActions){
    for(int i=0; i<numAgents; i++){
        assert(validAgentAction(i, agentActions[i]));
    }
    timer++;

    for(int i=0; i<numAgents; i++){
        Pos newHead = snakes[i].head.shift(agentActions[i]);
        setGridValue(snakes[i].head, agentActions[i] + 5*i);
        snakes[i].head = newHead;
        setGridValue(newHead, 4 + 5*i);
        if(newHead == apple){
            snakes[i].size ++;
            actionType = 1;
        }
        else{
            int tailDir = getGridValue(snakes[i].tail) - 5*i;
            setGridValue(snakes[i].tail, -1);
            snakes[i].tail = snakes[i].tail.shift(tailDir);
        }
    }
}

void Environment::chanceAction(int actionIndex){
    assert(validChanceAction(actionIndex));
    apple.x = actionIndex / boardy;
    apple.y = actionIndex % boardy;
    actionType = 0;
}


void Environment::inputSymmetric(Agent& net, int t, int activeAgent){
    networkInput* a = net.input;
    int m = boardx-1;
    int sym[8][2][3] = {
        {{ 1, 0, 0},{ 0, 1, 0}},
        {{ 0,-1, m},{ 1, 0, 0}},
        {{-1, 0, m},{ 0,-1, m}},
        {{ 0, 1, 0},{-1, 0, m}},
        {{ 0, 1, 0},{ 1, 0, 0}},
        {{ 1, 0, 0},{ 0,-1, m}},
        {{ 0,-1, m},{-1, 0, m}},
        {{-1, 0, m},{ 0, 1, 0}}
    };
    int x, y;
    
    // Input snake body:
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            x = sym[t][0][0]*i + sym[t][0][1]*j + sym[t][0][2];
            y = sym[t][1][0]*i + sym[t][1][1]*j + sym[t][1][2];
            if(grid[i][j] == -1){
                a->snake[x][y] = -1;
            }
            else{
                int agentID = grid[i][j] / 5;
                int dir = grid[i][j] % 5;

                int inc;
                if(agentID == activeAgent){
                    inc = 0;
                }
                else{
                    inc = 6;
                }
                a->snake[x][y] = (symDir[t][0]*dir + symDir[t][1] + 4) % 4 + inc;
            }
        }
    }

    // Input snake head, tail, and apple

    for(int i=0; i<numAgents; i++){
        int inc = -1;
        // Note: this only works for two-player.
        if(i == activeAgent){
            inc = 0;
        }
        else{
            inc = 6;
        }
        x = sym[t][0][0]*snakes[i].head.x + sym[t][0][1]*snakes[i].head.y + sym[t][0][2];
        y = sym[t][1][0]*snakes[i].head.x + sym[t][1][1]*snakes[i].head.y + sym[t][1][2];
        a->snake[x][y] = 4 + inc;
        x = sym[t][0][0]*snakes[i].tail.x + sym[t][0][1]*snakes[i].tail.y + sym[t][0][2];
        y = sym[t][1][0]*snakes[i].tail.x + sym[t][1][1]*snakes[i].tail.y + sym[t][1][2];
        a->snake[x][y] = 5 + inc;
    }
    x = sym[t][0][0]*apple.x + sym[t][0][1]*apple.y + sym[t][0][2];
    y = sym[t][1][0]*apple.x + sym[t][1][1]*apple.y + sym[t][1][2];
    a->snake[x][y] = 6*numAgents;
    
    // FILL IN VALID ACTIONS FOR NETWORK
    if(actionType == 0){
        for(int i=0; i<4; i++){
            net.validAction[(symDir[t][0]*i + symDir[t][1] + 4) % 4] = validAgentAction(activeAgent, i);
        }
    }

    // // logging agent's input for debugging:
    // for(int i=0; i<boardx; i++){
    //     for(int j=0; j<boardy; j++){
    //         cout<<a->snake[i][j]<<' ';
    //     }
    //     cout<<'\n';
    // }
    // for(int i=0; i<4; i++){
    //     cout<<net.validAction[i]<<' ';
    // }
    // cout<<'\n';
}


/*
void Environment::copyEnv(Environment* e){
    timer = e->timer;
    actionType = e->actionType;
    snakeSize = e->snakeSize;
    headx = e->headx;
    heady = e->heady;
    tailx = e->tailx;
    taily = e->taily;
    applex = e->applex;
    appley = e->appley;
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            snake[i][j] = e->snake[i][j];
        }
    }
}*/

void Environment::print(){ // optional function for debugging
    /*
    ofstream fout(outAddress, ios::app);
    fout<<"Timer: "<<timer<<'\n';
    fout<<"Score: "<<score<<'\n';
    fout<<"Action type: "<<actionType<<'\n';
    fout<<"Snake size: "<<snakeSize<<'\n';
    int i,j;
    for(i=0; i<boardx; i++){
        for(j=0; j<boardy; j++){
            if(i == applex && j == appley){
                fout<<'A'<<' ';
            }
            else{
                if(snake[i][j] == -1){
                    fout<<". ";
                }
                else{
                    fout<<snake[i][j]<<' ';
                }
            }
        }
        fout<<'\n';
    }
    fout.close();
    */
}

void Environment::log(string outFile){ // optional function for debugging
    ofstream fout(outFile, ios::app);
    fout<<"Timer: "<<timer<<'\n';
    fout<<"Action type: "<<actionType<<'\n';
    fout<<"Snake sizes: ";
    for(int i=0; i<numAgents; i++){
        fout<<snakes[i].size<<' ';
    }
    fout<<'\n';
    char output[2*boardx+1][2*boardy+1];
    for(int i=0; i<2*boardx+1; i++){
        for(int j=0; j<2*boardy+1; j++){
            output[i][j] = ' ';
        }
    }
    for(int i=0; i<2*boardx+1; i++){
        output[i][0] = '#';
        output[i][2*boardy] = '#';
    }
    for(int j=0; j<2*boardy+1; j++){
        output[0][j] = '#';
        output[2*boardx][j] = '#';
    }
    char body[numAgents] = {'x', 'y'};
    char head[numAgents] = {'X', 'Y'};
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            int val = getGridValue(Pos(i, j));
            int snakeID = val / 5;
            int d = val % 5;
            char out;
            if(apple == Pos(i, j)){
                out = 'A';
            }
            else if(val == -1){
                out = '.';
            }
            else if(d == 4){
                out = head[snakeID];
            }
            else{
                out = body[snakeID];
                char bar;
                if(d%2 == 0) bar = '-';
                else bar = '|';
                output[2*i+1 + dir[d][0]][2*j+1 + dir[d][1]] = bar;
            }
            output[2*i+1][2*j+1] = out;
        }
    }
    for(int i=0; i<2*boardx+1; i++){
        for(int j=0; j<2*boardy+1; j++){
            fout<<output[i][j];
        }
        fout<<'\n';
    }
    fout<<'\n';
}

void Environment::computeRewards(){
    for(int i=0; i<numAgents; i++){
        rewards[i] = 0;
    }
    for(int i=0; i<numAgents; i++){
        bool surrounded = true;
        for(int d=0; d<4; d++){
            if(validAgentAction(i, d)){
                surrounded = false;
                break;
            }
        }
        if(surrounded) rewards[i] -= 20;
    }
    for(int i=0; i<numAgents; i++){
        if(apple == snakes[i].head){
            rewards[i] ++;
        }
    }
    // zero-sum
    int diff = rewards[0] - rewards[1];
    rewards[0] = diff;
    rewards[1] = -diff;
    /*
    if(snakeSize == boardx * boardy) return 10;
    if(timer == maxTime) return 0;
    if(isEndState()) return -10;
    if(actionType == 1) return 1;
    return 0;*/
}

bool operator == (const Environment& e1, const Environment& e2){
    //if(e1.timer != e2.timer) return false;
    if(e1.actionType != e2.actionType) return false;
    for(int i=0; i<numAgents; i++){
        if(e1.snakes[i] != e2.snakes[i]) return false;
    }
    if(e1.apple != e2.apple) return false;
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            if(e1.grid[i][j] != e2.grid[i][j]) return false;
        }
    }
    return true;
}