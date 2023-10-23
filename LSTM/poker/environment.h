
#include "lstm.h"
#include <utility>
#include <algorithm>
#include <random>

#ifndef environment_h
#define environment_h

#define numSuits 4
#define numVals 13
#define deckSize (numSuits * numVals) // = 52

#define TIME_HORIZON 5
#define NUM_ACTIONS deckSize
#define DISCOUNT_FACTOR 1

#define inputSize (2 * deckSize)

/*
Train settings

double initParam = 0.1;
double learnRate = 0.01;
double momentum = 0.7;
int batchSize = 20;
double regRate = 0.00001;
double meanUpdate = 0.001;

PG(){
    Model m(inputSize);
    m.addLSTM(10);
    m.addOutput(NUM_ACTIONS);
    seq = ModelSeq(m, TIME_HORIZON, initParam);
}

Train for 300000 steps, nonparallel

*/

#define STRAIGHT_FLUSH "STRAIGHT_FLUSH"
#define QUAD "QUAD"
#define FULL_HOUSE "FULL_HOUSE"
#define FLUSH "FLUSH"
#define STRAIGHT "STRAIGHT"
#define TRIP "TRIP"
#define TWO_PAIR "TWO_PAIR"
#define PAIR "PAIR"
#define HIGH "HIGH"

const vector<int> straight_order{12,0,1,2,3,4,5,6,7,8,9,10,11,12};

const int numStrengths = 9;
const string strength_order[9] = {
    STRAIGHT_FLUSH, QUAD, FULL_HOUSE,
    FLUSH, STRAIGHT, TRIP,
    TWO_PAIR, PAIR, HIGH
};

class Strength{
public:
    string type;
    vector<int> vals;

    Strength(string type_, int val_){
        type = type_;
        vals = vector<int>{val_};
    }

    Strength(string type_, vector<int> vals_){
        type = type_;
        for(auto v : vals_){
            vals.push_back(v);
        }
    }

    string toString(){
        string s = type;
        for(auto v : vals){
            s += ' ' + to_string(v);
        }
        return s;
    }

    int strengthIndex(){
        for(int i=0; i<numStrengths; i++){
            if(strength_order[i] == type){
                return i;
            }
        }
        assert(false);
        return -1;
    }

    bool compare(Strength other){ // returns true if this is stronger than other
        int s = strengthIndex();
        int so = other.strengthIndex();
        if(s != so) return s < so;
        assert(vals.size() == other.vals.size());
        for(int i=0; i<vals.size(); i++){
            if(vals[i] != other.vals[i]) return vals[i] > other.vals[i];
        }
        return false;
    }
};

class Hand{
private:
    int counts[numVals];
    vector<pair<int, int> > sortedCounts;

    void getCounts();
    int straightFlushCheck();
    int quadCheck();
    int fullCheck();
    int flushCheck();
    int straightCheck();
    int tripCheck();
    vector<int> twoCheck();
    vector<int> pairCheck();
    vector<int> highCheck();

public:
    bool cards[numSuits][numVals];

    Hand();
    string toString();

    Strength getStrength();
    void addCard(int cardID);
    bool checkCard(int cardID);
};

class Environment{
public: // CHANGED TO PUBLIC
    Hand agentHand;
    Hand oppHand;
    vector<int> cards;
    int cardIndex;

public:
    int time;
    bool endState;

    Environment();
    string toString();

    vector<int> validActions();
    double makeAction(int* actions); // returns reward

    void inputObservations(Data* input);
};

#endif