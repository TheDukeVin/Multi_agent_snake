
/*

g++ -O2 -std=c++17 -pthread -I "/Users/kevindu/Desktop/Employment/Multiagent Snake Research/multiagent_snake/LSTM" main.cpp modelseq.cpp model.cpp layer.cpp layers/lstm.cpp layers/policy.cpp layers/conv.cpp layers/pool.cpp params.cpp node.cpp PG.cpp common.cpp rabbits/environment.cpp

g++ -O2 -std=c++17 -pthread -I /n/home04/kevindu/MultiagentSnake/LSTM main.cpp modelseq.cpp model.cpp layer.cpp layers/lstm.cpp layers/policy.cpp layers/conv.cpp layers/pool.cpp params.cpp node.cpp PG.cpp common.cpp rabbits/environment.cpp && sbatch lstm.slurm

*/

#include "lstm.h"
// #include "token/environment.h"
// #include "dice/environment.h"
#include "rabbits/environment.h"
// #include "bin/environment.h"
// #include "poker/environment.h"
#include <iomanip>

#ifndef PG_h
#define PG_h

#define SINGLE_ACT 0
#define MULT_ACT 1

class PG{
public:
    ModelSeq seq;

    double reward[TIME_HORIZON];

    double valueMean[TIME_HORIZON];

    int MODE = SINGLE_ACT;

    double initParam = 0.1;
    double learnRate = 0.005;
    double momentum = 0.7;
    int batchSize = 20;
    int numSubThreads = 20;
    double regRate = 0.00001;
    double meanUpdate = 0.001;

    PG(){
        Model m(Shape(2*numRabbits, boardSize, boardSize));
        m.addConv(Shape(6, 5, 5), 3, 3);
        m.addConv(Shape(20, 4, 4), 2, 2);
        m.addPool(Shape(20, 2, 2));
        m.addLSTM(80);
        m.addOutput(NUM_ACTIONS);
        // m.addLSTM(200);
        // m.addOutput(NUM_ACTIONS);
        seq = ModelSeq(m, TIME_HORIZON, initParam);
    }

    string fileOut;
    double rollOutRewardSum;
    double finalReward;

    void rollOut(bool printGame = false); // returns cumulative reward
    void multRollOut(int numRolls);
    void computeSoftmax(double* weights, double* policy, vector<int> validActions);
    void computeSigmoid(double* weights, double* policy, vector<int> validActions); // for multiple action
    void train();

    // divides batches evenly among threads, each thread
    // receiving batchSize/numSubThreads rollouts.
    void trainParallel(int evalPeriod, int numIter);
    void setLearnRate(double lr);
};

#endif