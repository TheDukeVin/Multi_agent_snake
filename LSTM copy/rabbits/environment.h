
#include "lstm.h"

#ifndef environment_h
#define environment_h

#define boardSize 5
#define boardArea (boardSize * boardSize)
#define numRabbits 3

#define TIME_HORIZON 100
#define NUM_ACTIONS (numRabbits * boardSize * boardSize)
#define DISCOUNT_FACTOR 0.98

#define inputSize (2 * numRabbits * boardSize * boardSize)

const int dir[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};

/*
Best training settings:

Model m(inputSize);
m.addLSTM(100);
m.addOutput(NUM_ACTIONS);

double initParam = 0.1;
double learnRate = 0.01;
double momentum = 0.7;
int batchSize = 20;
double regRate = 0.00001;
double meanUpdate = 0.001;

train for 100000 steps, parallel
*/

class Pos{
public:
    int x, y;

    Pos(){
        x = y = -1;
    }

    Pos(int index){
        x = index / boardSize;
        y = index % boardSize;
    }

    Pos(int _x, int _y){
        x = _x; y = _y;
    }

    bool inBounds(){
        return 0 <= x && x < boardSize && 0 <= y && y < boardSize;
    }

    int toIndex(){
        return x * boardSize + y;
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
    Pos rabbits[numRabbits];
    Pos carrot;
    int carrotType;

    void placeCarrot();

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