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

    ofstream fout1(valueLossFile);
    fout1.close();
    ofstream fout2(policyLossFile);
    fout2.close();
    ofstream fout3(valueNormFile);
    fout3.close();
    ofstream fout4(policyNormFile);
    fout4.close();
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

void DataQueue::backPropRollout(LSTM::PVUnit& agent, int rolloutIndex){
    double policy[2*maxTime][numAgentActions];
    for(int i=0; i<queue[rolloutIndex].size(); i++){
        units[i]->copyParams(&agent);
        int symID = 0;
        Environment env = queue[rolloutIndex][i].e;
        // BUG:
        // env.inputSymmetric(a, symID, TRAIN_ACTIVE);
        env.inputSymmetric(*units[i], symID, TRAIN_ACTIVE);
        units[i]->forwardPass();
        computeSoftmaxPolicy(units[i]->policyOutput->data, numAgentActions, env.validAgentActions(TRAIN_ACTIVE), policy[i]);
    }
    // ofstream dataOut("data.out", ios::app);
    for(int i=queue[rolloutIndex].size()-1; i>=0; i--){
        // Calculate policy gradients
        units[i]->resetGradient();
        for(int j=0; j<agent.policyBranch->outputSize; j++){
            units[i]->policyOutput->gradient[j] = 0;
        }
        units[i]->valueOutput->gradient[0] = 0;

        Environment env = queue[rolloutIndex][i].e;
        // dataOut << env.toString() << '\n';
        if(!env.isEndState() && env.actionType == 0){
            for(auto a : env.validAgentActions(TRAIN_ACTIVE)){
                units[i]->policyOutput->gradient[a] = policy[i][a] - queue[rolloutIndex][i].expectedPolicy[a];
                // dataOut << "Policy: " << policy[i][a] << ' ' << queue[rolloutIndex][i].expectedPolicy[a] << '\n';
                policyLoss -= queue[rolloutIndex][i].expectedPolicy[a] * log(policy[i][a]);
            }
        }
        units[i]->backwardPass();
        agent.accumulateGradient(units[i]);

        // Update norm of policy gradients
        for(int j=0; j<units[i]->allBranches.size(); j++){
            for(int k=0; k<units[i]->allBranches[j]->layers.size(); k++){
                for(int l=0; l<units[i]->allBranches[j]->layers[k]->params->size; l++){
                    policyGradNorm += pow(units[i]->allBranches[j]->layers[k]->params->gradient[l], 2);
                }
            }
        }

        // Calculate value gradients
        units[i]->resetGradient();
        for(int j=0; j<agent.policyBranch->outputSize; j++){
            units[i]->policyOutput->gradient[j] = 0;
        }
        units[i]->valueOutput->gradient[0] = units[i]->valueOutput->data[0] - queue[rolloutIndex][i].expectedValue;
        // dataOut << "Value: " << units[i]->valueOutput->data[0] << ' ' << queue[rolloutIndex][i].expectedValue << '\n';
        units[i]->backwardPass();
        agent.accumulateGradient(units[i]);
        valueLoss += pow(units[i]->valueOutput->gradient[0], 2);

        // Track norm of gradients
        for(int j=0; j<units[i]->allBranches.size(); j++){
            for(int k=0; k<units[i]->allBranches[j]->layers.size(); k++){
                for(int l=0; l<units[i]->allBranches[j]->layers[k]->params->size; l++){
                    valueGradNorm += pow(units[i]->allBranches[j]->layers[k]->params->gradient[l], 2);
                }
            }
        }
    }
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

void DataQueue::empty(){
    index = 0;
    numFilled = 0;
    for(int i=0; i<queueSize; i++){
        queue[i] = vector<Data>();
    }
}

void DataQueue::trainAll(LSTM::PVUnit& a){
    int i,j;
    ofstream valueLossOut(valueLossFile, ios::app);
    ofstream policyLossOut(policyLossFile, ios::app);
    ofstream valueNormOut(valueNormFile, ios::app);
    ofstream policyNormOut(policyNormFile, ios::app);

    double valSum = 0;
    double polSum = 0;
    for(i=0; i<numFilled; i++){
        valueLoss = policyLoss = 0;
        valueGradNorm = policyGradNorm = 0;

        backPropRollout(a, i);

        // valSum += valueLoss;
        // polSum += policyLoss;
        valueLossOut << valueLoss << ',';
        policyLossOut << policyLoss << ',';
        valueNormOut << valueGradNorm << ',';
        policyNormOut << policyGradNorm << ',';
    }
    valueLossOut << '\n';
    policyLossOut << '\n';
    valueNormOut << '\n';
    policyNormOut << '\n';
    a.updateParams(learnRate, momentum, 0.0001);
}