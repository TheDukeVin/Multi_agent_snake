
#include "lstm.h"

#ifndef environment_h
#define environment_h

#define TIME_HORIZON 100
#define NUM_ACTIONS 4
#define DISCOUNT_FACTOR 0.98

#define boardSize 10
#define proximity 1
#define inputSize (boardSize * 4)

// for 5x5 board, proximity 1
// for 10x10 board, proximity 2

// train for 200000 steps
// 50 step horizon, discount factor 1

// for 10x10 board, proximity 1

// Train for 500000 steps
// 100 step horizon, discount factor 0.98

// double initParam = 0.1;
// double learnRate = 0.02;
// double momentum = 0.9;
// int batchSize = 60;
// double meanUpdate = 0.0001;

// Model m(inputSize);
// m.addLSTM(20);
// m.addOutput(NUM_ACTIONS);

const int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Pos{
public:
    int x, y;

    Pos(){
        x = y = -1;
    }

    Pos(int _x, int _y){
        x = _x; y = _y;
    }

    bool inBounds(){
        return 0 <= x && x < boardSize && 0 <= y && y < boardSize;
    }

    Pos shift(int d){
        return Pos(x + dir[d][0], y + dir[d][1]);
    }

    friend bool operator == (const Pos& p, const Pos& q){
        return (p.x == q.x) && (p.y == q.y);
    }

    friend bool operator != (const Pos& p, const Pos& q){
        return (p.x != q.x) || (p.y != q.y);
    }
};

class Environment{
private:
    Pos runner;
    Pos token;

    void placeToken();

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