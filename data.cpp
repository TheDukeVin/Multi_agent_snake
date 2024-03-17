//
//  data.cpp
//  data
//
//  Created by Kevin Du on 2/2/22.
//

#include "snake.h"

Data::Data(Environment* givenEnv, double givenExpected){
    e = *givenEnv;
    expectedValue = givenExpected;
}

DataQueue::DataQueue(LSTM::PVUnit* structure){
    index = 0;
    numFilled = 0;
    for(int i=0; i<queueSize; i++){
        queue[i] = vector<Data>();
    }
    for(int i=0; i<maxTime*2; i++){
        LSTM::PVUnit* prevUnit;
        if(i>0) prevUnit = units[i-1];
        else prevUnit = NULL;

        units[i] = new LSTM::PVUnit(structure, prevUnit);
    }
}

void DataQueue::enqueue(vector<Data> d){
    assert(index < currSize);
    queue[index] = d;
    numFilled = max(index + 1, numFilled);
    index++;
    if(index == currSize){
        index = 0;
    }
}

void DataQueue::backPropRollout(LSTM::PVUnit& a, int rolloutIndex){
    double policy[2*maxTime][numAgentActions];
    valueLoss = policyLoss = 0;
    for(int i=0; i<queue[rolloutIndex].size(); i++){
        units[i]->copyParams(&a);
        int symID = 0;
        Environment env = queue[rolloutIndex][i].e;
        env.inputSymmetric(a, symID, TRAIN_ACTIVE);
        units[i]->forwardPass();
        computeSoftmaxPolicy(units[i]->policyOutput->data, numAgentActions, env.validAgentActions(TRAIN_ACTIVE), policy[i]);
    }
    // ofstream dataOut("data.out", ios::app);
    for(int i=queue[rolloutIndex].size()-1; i>=0; i--){
        for(int j=0; j<a.policyBranch->outputSize; j++){
            units[i]->policyOutput->gradient[j] = 0;
        }
        Environment env = queue[rolloutIndex][i].e;
        // dataOut << env.toString() << '\n';
        if(!env.isEndState() && env.actionType == 0){
            for(auto a : env.validAgentActions(TRAIN_ACTIVE)){
                units[i]->policyOutput->gradient[a] = policy[i][a] - queue[rolloutIndex][i].expectedPolicy[a];
                // dataOut << "Policy: " << policy[i][a] << ' ' << queue[rolloutIndex][i].expectedPolicy[a] << '\n';
                policyLoss -= queue[rolloutIndex][i].expectedPolicy[a] * log(policy[i][a]);
            }
        }
        units[i]->valueOutput->gradient[0] = units[i]->valueOutput->data[0] - queue[rolloutIndex][i].expectedValue;
        units[i]->backwardPass();
        // dataOut << "Value: " << units[i]->valueOutput->data[0] << ' ' << queue[rolloutIndex][i].expectedValue << '\n';
        a.accumulateGradient(units[i]);
        valueLoss += pow(units[i]->valueOutput->gradient[0], 2);
    }
}

void DataQueue::trainAgent(LSTM::PVUnit& a, string outFile){
    int i,j;
    string valueLossOut;
    string policyLossOut;
    string normOut;
    for(i=0; i<numBatches; i++){
        backPropRollout(a, rand() % numFilled);
        a.updateParams(learnRate / maxTime, momentum, 0.0001);

        // Track norm of gradients
        // double gradNorm = 0;
        // for(int j=0; j<a.allBranches.size(); j++){
        //     for(int k=0; k<a.allBranches[j]->layers.size(); k++){
        //         for(int l=0; l<a.allBranches[j]->layers[k]->params->size; l++){
        //             gradNorm += pow(a.allBranches[j]->layers[k]->params->gradient[l], 2);
        //         }
        //     }
        // }

        // if(i > 0){
        //     valueLossOut += ",";
        //     policyLossOut += ",";
        //     normOut += ",";
        // }
        // valueLossOut += to_string(valueLoss);
        // policyLossOut += to_string(policyLoss);
        // normOut += to_string(gradNorm);

        // for(j=0; j<batchSize; j++){
        //     int gameIndex = rand() % numFilled;
        //     queue[gameIndex][rand() % gameLengths[gameIndex]].trainAgent(a);
        // }
        // a.updateParameters(learnRate / batchSize, momentum);
    }
    ofstream fout(outFile);
    fout << valueLossOut << '\n';
    fout << policyLossOut << '\n';
    fout << normOut << '\n';
    fout.close();
}

vector<int> DataQueue::readGames(string fileName){
    return vector<int>();
    // ifstream fin (fileName);
    // vector<int> scores;
    // string s;
    // bool nextLine = false;
    // while(getline(fin, s)){
    //     if (s[0] == 'G'){
    //         nextLine = true;
    //         continue;
    //     }
    //     if(!nextLine) continue;
    //     stringstream sin (s);
    //     int input;
    //     sin >> input;

    //     vector<Environment> envs;
    //     Environment initialEnv;
    //     initialEnv.initialize();
    //     initialEnv.chanceAction(input);
    //     envs.push_back(initialEnv);
        
    //     for(int i=1; true; i++){
    //         sin>>input;
    //         Environment new_env = envs[i-1];
    //         Action chosenAction(envs[i-1].actionType, input);
    //         new_env.makeAction(chosenAction);
            
    //         envs.push_back(new_env);
    //         if(new_env.isEndState()){
    //             break;
    //         }
    //     }
    //     int gameLength = envs.size();
        
    //     Data* game = new Data[gameLength];
        
    //     for(int i=gameLength-1; i>=0; i--){
    //         game[i].e = envs[i];
    //     }
    //     enqueue(game, gameLength);
    //     scores.push_back(0);

    //     nextLine = false;
    // }
    // return scores;
}
