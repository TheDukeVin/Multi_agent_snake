//
//  snake.h
//  snake
//
//  Created by Kevin Du on 1/18/22.
//

#include <iostream>
#include <fstream>
#include <thread>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <string>
#include <vector>
#include <list>
#include <cassert>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <stdio.h>

#ifndef snake_h
#define snake_h

using namespace std;

// environment details

#define numAgents 2

#define boardx 10
#define boardy 10
#define maxTime 1000

#define numAgentActions 4
#define numChanceActions (boardx * boardy)
#define maxNumActions (boardx * boardy)

#define discountFactor 0.98

// network deatils

#define maxNorm 100
#define batchSize 100
#define numBatches 30

#define queueSize 10000

// training details

#define maxStates (maxTime * 2 * numPaths)
#define maxAgentQueue 30
#define evalPeriod 400
#define numEvalGames 20

#define scoreNorm 20

#define numGames ((maxAgentQueue - 1) * evalPeriod)
#define numPaths 200


// Passing Value or Full (or Policy)

#define PASS_VALUE 0
#define PASS_FULL 1
#define PASS_POLICY 2

// Multithreading

#define NUM_THREADS 50

// Miscellaneous

#define inf 1e+06

const string outAddress = "snake_conv.txt";

double squ(double x);

int max(int x, int y);

double min(double x, double y);

double abs(double x);

// take a random sample from a distribution on 0,...,N-1
// -1 represents an invalid value.
int sampleDist(double* dist, int N);

// For the network
double randWeight(double startingParameterRange);

double nonlinear(double x);

// If f = nonlinear, then this function is (f' \circ f^{-1}).
double dinvnonlinear(double x);


class Nash{
public:
    int N, M;
    double** A;

    const double rate = 1e-01;
    const double alpha = 1;

    double* p1;
    double* p2;

    double* grad1;
    double* grad2;

    double* next1;
    double* next2;

    Nash(){}

    Nash(int N_, int M_){
        initialize(N_, M_);
    }
    void find_equilibrium(int iter, double threshold);
    double exploitabilty();
    void evaluate(int size, int iter, int num_trial);

private:
    void initialize(int N_, int M_);
    void compute_gradients(double* p1, double* p2);
    void compute_step(double* p1, double* p2, double step_size);
    void exp_step(double* policy, double* grad, double* next_policy, double step_size, int size);
    void check_policy(double* policy, int size);
};


// Input to the network


struct networkInput{
    // 0, 1, 2, 3 = snake units pointing to next unit.
    // 4 = head of snake
    // 5 = tail of snake
    // +6k = agent k
    // 6*numAgents = the apple.
    int snake[boardx][boardy];
};

const int symDir[8][2] = {
    { 1,0},
    { 1,3},
    { 1,2},
    { 1,1},
    {-1,1},
    {-1,2},
    {-1,3},
    {-1,0}
};

class Layer{
public:
    ifstream* netIn;
    ofstream* netOut;
    int numParams, numWeights, numBias;
    double* params;
    double* weights;
    double* bias;
    double* Dparams;
    double* Dweights;
    double* Dbias;
    
    double* inputs;
    double* outputs;
    double* Dinputs;
    double* Doutputs;
    
    virtual void pass(){};
    virtual void backProp(bool increment = false){};
    virtual void accumulateGradient(){};
    void setupParams();
    void randomize(double startingParameterRange);
    void resetGradient();
    void updateParameters(double mult, double momentum);
    void save();
    void readNet();
    
    virtual ~Layer(){}
};

class ConvLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int w1, w2, w3;
    
    ConvLayer(int inD, int inH, int inW, int outD, int outH, int outW, int convH, int convW);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    virtual void accumulateGradient();
    
    virtual ~ConvLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class PoolLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int* maxIndices;
    
    PoolLayer(int inD, int inH, int inW, int outD, int outH, int outW);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    
    virtual ~PoolLayer(){
        delete[] maxIndices;
    }
};

class DenseLayer : public Layer{
public:
    int inputSize, outputSize;
    
    DenseLayer(int inSize, int outSize);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    virtual void accumulateGradient();
    
    virtual ~DenseLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

// Input layer tailored to snake environment

class InputLayer : public Layer{
public:
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int posShiftr, posShiftc;
    int w1, w2, w3;
    
    networkInput* env;
    
    InputLayer(int outD, int outH, int outW, int convH, int convW, networkInput* input);
    
    virtual void pass();
    virtual void accumulateGradient();
    
    virtual ~InputLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class OutputLayer : public Layer{
public:
    int inputSize, outputSize;
    
    OutputLayer(int inSize, int outputSize);
    
    virtual void pass();
    virtual void backProp(bool increment = false);
    virtual void accumulateGradient();
    
    virtual ~OutputLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class Branch{
public:
    int numLayers;
    
    // For network initiation
    int prevDepth, prevHeight, prevWidth;
    networkInput* input;
    vector<Layer*> layerHold;
    Layer** layers;
    
    double* prevActivation;
    double* prevDbias;
    double* output;
    double* Doutput;
    
    void initEnvironmentInput(int depth, int height, int width, int convHeight, int convWidth);
    void addConvLayer(int depth, int height, int width, int convHeight, int convWidth);
    void addPoolLayer(int depth, int height, int width);
    void addFullyConnectedLayer(int numNodes);
    void addOutputLayer(int numNodes);
    void setup();
};

class Agent{
public:
    networkInput* input;
    int numLayers;
    
    Layer** layers; // keep an array of pointers, since derived classes need to be accessed by reference.
    Branch commonBranch;
    Branch policyBranch;
    Branch valueBranch;
    
    double policyOutput[numAgentActions];
    double valueOutput;
    bool validAction[numAgentActions]; // MUST BE FILLED IN for evaluation to work.
    double policyExpected[numAgentActions];
    double valueExpected;
    
    // For file I/O
    ifstream* netIn;
    ofstream* netOut;
    
    Agent(){
        input = new networkInput;
        commonBranch.input = input;
    }
    
    // For network usage and training
    void copyParam(Agent& a);
    void setupCommonBranch();
    void setup();
    void resetGradient();
    void randomize(double startingParameterRange);
    
    void pass(int mode); // Inputs are PASS_VALUE or PASS_FULL
    void backProp(int mode);
    void updateParameters(double mult, double momentum);
    
    void save(string fileName);
    void readNet(string fileName);
};


// Environment things

const int numActions[2] = {numAgentActions, numChanceActions};

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
        return 0 <= x && x < boardx && 0 <= y && y < boardy;
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

class Snake{
public:
    int size;
    Pos head;
    Pos tail;

    Snake(){
        size = -1;
    }

    friend bool operator != (const Snake& r, const Snake& s){
        return (r.size != s.size) || (r.head != s.head) || (r.tail != s.tail);
    }
};

// Class to encode action info at a time step
class Action{
public:
    int actionType;
    int agentActions[numAgents];
    int chanceAction;

    int actionID(){
        if(actionType == 0) return agentActions[0] + agentActions[1] * numAgentActions;
        return chanceAction;
    }
};

class Environment{
public:
    int timer;
    int actionType; // 0 = action state, 1 = reaction state.

    Snake snakes[numAgents];
    Pos apple;

     // -1 = not snake. 0 to 3 = snake unit pointing to next unit. 4 = head.
     // +5k for agent k
    int grid[boardx][boardy];

    void setGridValue(Pos p, int val);
    int getGridValue(Pos p);
    
    double rewards[numAgents];
    
    void initialize();
    
    bool isEndState();

    bool validAgentAction(int agentID, int action); // from agentActions array
    bool validChanceAction(int pos);

    void makeAction(Action chosenAction);
    // void setAgentAction(int agentID, int action);
    void agentAction(int* agentActions);
    void chanceAction(int actionIndex);

    //void setAction(Environment* currState, int actionIndex);
    void inputSymmetric(Agent& net, int t, int activeAgent);
    //void copyEnv(Environment* e);
    void print();// optional function for debugging
    void log(string outFile);// optional function for debugging
    
    void computeRewards();

    friend bool operator == (const Environment& e1, const Environment& e2);

    /*
    double features(int featureType);
    double features2(Environment* nextState, int featureType);
    double distance(int featureType);

    void BFS(int* dist, int sourcex, int sourcey, bool fill = true);
    int lastAction();
    */
};

class EnvHash{
public:
    size_t operator()(const Environment& env) const {
        int M = 10000019;
        int val = 0;
        //val = (val * 3 + env.timer) % M;
        val = (val * 3 + env.actionType) % M;
        val = (val * 3 + env.apple.x) % M;
        val = (val * 3 + env.apple.y) % M;
        for(int i=0; i<boardx; i++){
            for(int j=0; j<boardy; j++){
                val = (val * 3 + env.grid[i][j]) % M;
            }
        }
        return val;
    }
};

// Data things

class Data{
public:
    int ACTIVE_AGENT = 0;
    int ADVERSARY_AGENT = 1;

    Environment e;
    double expectedValue;
    double expectedPolicy[numAgentActions];
    
    Data(){}
    Data(Environment* givenEnv, double givenExpected);
    void trainAgent(Agent& a);
};

class DataQueue{
public:
    Data* queue[queueSize];
    int gameLengths[queueSize];
    int index;
    int currSize, numFilled;
    double learnRate, momentum;
    
    DataQueue();
    void enqueue(Data* d, int gameLength);
    void trainAgent(Agent& a);
    vector<int> readGames(); // returns the maximum score out of the games read.
};

// Trainer

const int TRAIN_ACTIVE = 0;
const int TRAIN_ADVERSARY = 1;

const int TRAIN_MODE = 0;
const int TEST_MODE = 1;

class MCTSModel{
public:
    int ACTIVE_AGENT;
    int ADVERSARY_AGENT;
    double actionTemperature = 2;
    double cUCB = 0.5;

    // MCTS model used to select moves for agent ACTIVE_AGENT.
    // Same model used to predict policy/value for both agents in the tree search
    Agent a;
    
    MCTSModel(){
        for(int i=0; i<maxStates; i++){
            outcomes[i] = NULL;
        }
    }
    
    //Storage for the tree:
    int* outcomes[maxStates];

    double* actionSums[maxStates];
    int* actionCounts[maxStates];

    int subtreeSize[maxStates];
    double sumScore[maxStates];
    double values[maxStates];

    double policy[numAgents][maxStates][numAgentActions];
    // double adversary_policy[maxStates][numAgentActions];
    
    // Implementing the tree search
    int index;

    int rootIndex;
    Environment rootEnv;
    
    // For executing a training iteration:
    double actionProbs[numAgentActions];
    
    void initializeNode(Environment& env, int currNode);
    
    int path[maxStates];
    int pathActions[maxStates];
    double rewards[maxTime*2];
    int times[maxTime*2];

    // int getAdversaryAction(int currIndex);

    void simulateAction(Environment& env, Action chosenAction);
    
    void expandPath();
    void printTree();
    void computeActionProbs();
    int optActionProbs();
    //int sampleActionProbs();
};

class Trainer{
public:
    Agent a;

    // test/train this agent against a competitor
    Agent competitor;

    int output_gameLength;
    Data* output_game;
    double total_reward; // for the active player.
    
    string gameLog;
    string valueLog;
    string valueOutput;

    // In models[0], ACTIVE_AGENT = 0 and ADVERSARY_AGENT = 1
    // In models[1], ACTIVE_AGENT = 1 and ADVERSARY_AGENT = 0
    MCTSModel models[numAgents];
    
    Trainer(){
        models[TRAIN_ACTIVE].ACTIVE_AGENT = TRAIN_ACTIVE;
        models[TRAIN_ACTIVE].ADVERSARY_AGENT = TRAIN_ADVERSARY;
        models[TRAIN_ADVERSARY].ACTIVE_AGENT = TRAIN_ADVERSARY;
        models[TRAIN_ADVERSARY].ADVERSARY_AGENT = TRAIN_ACTIVE;
    };
    
    Environment roots[maxTime*2];
    int rootIndices[maxTime*2];

    void trainGame(int mode);

    int getRandomChanceAction(Environment* e);
};

#endif /* snake_h */
