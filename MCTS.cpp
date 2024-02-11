//
//  MCTS.cpp
//  MCTS
//
//  Created by Kevin Du on 2/24/23.
//

#include "snake.h"


void MCTSModel::initializeNode(Environment& env, int currNode){
    if(outcomes[currNode] != NULL){
        delete outcomes[currNode];
    }
    if(actionCounts[currNode] != NULL){
        delete actionCounts[currNode];
    }
    if(actionSums[currNode] != NULL){
        delete actionSums[currNode];
    }
    if(env.actionType == 0){
        outcomes[currNode] = new int[numAgentActions * numAgentActions];
        for(int i=0; i<numAgentActions; i++){
            for(int j=0; j<numAgentActions; j++){
                Action act;
                act.actionType = 0;
                act.agentActions[ACTIVE_AGENT] = i;
                act.agentActions[ADVERSARY_AGENT] = j;
                if(env.validAgentAction(ACTIVE_AGENT, i) && env.validAgentAction(ADVERSARY_AGENT, j)){
                    outcomes[currNode][act.actionID()] = -1;
                }
                else{
                    outcomes[currNode][act.actionID()] = -2;
                }
            }
        }
    }
    else{
        outcomes[currNode] = new int[numChanceActions];
        for(int i=0; i<numChanceActions; i++){
            if(env.validChanceAction(i)){
                outcomes[currNode][i] = -1;
            }
            else{
                outcomes[currNode][i] = -2;
            }
        }
    }
    subtreeSize[currNode] = 0;
    sumScore[currNode] = 0;
    if(env.actionType == 0){
        actionCounts[currNode] = new int[numAgentActions];
        actionSums[currNode] = new double[numAgentActions];
        for(int i=0; i<numAgentActions; i++){
            if(env.validAgentAction(ACTIVE_AGENT, i)){
                actionCounts[currNode][i] = 0;
                actionSums[currNode][i] = 0;
            }
            else{
                actionCounts[currNode][i] = -1;
            }
        }
    }
    else{
        actionCounts[currNode] = new int[numChanceActions];
        actionSums[currNode] = new double[numChanceActions];
        for(int i=0; i<numChanceActions; i++){
            if(env.validChanceAction(i)){
                actionCounts[currNode][i] = 0;
                actionSums[currNode][i] = 0;
            }
            else{
                actionCounts[currNode][i] = -1;
            }
        }
    }

    // Use learned features

//    values[currNode] = 3 + env.getReward()
//                         + ((env.features(2) - 1) * 4)
//                         - (env.features(3) * 0.1)
//                         + (1 - env.features(5));
//
//    if(env.actionType == 0){
//        int numValidActions = 0;
//        for(int d=0; d<numAgentActions; d++){
//            numValidActions ++;
//        }
//        for(int d=0; d<numAgentActions; d++){
//            if(outcomes[currNode][d] != -2){
//                policy[currNode][d] = 1.0 / numValidActions;
//            }
//            else{
//                policy[currNode][d] = -1;
//            }
//        }
//    }
}

void MCTSModel::evaluateEnv(Environment& env, int currNode, LSTM::PVUnit* currUnit){
    
    // Evaluate the network at the current node.

    double pred_values[numAgents];
    
    for(int m=0; m<numAgents; m++){
        // int symID = rand()%8;
        int symID = 0;
        
        env.inputSymmetric(*currUnit, symID, m);
        currUnit->forwardPass();
        if(currNode == -1) continue;
        pred_values[m] = currUnit->valueOutput->data[0];
        if(env.actionType == 0){
            computeSoftmaxPolicy(currUnit->policyOutput->data, numAgentActions, env.validAgentActions(m), policy[m][currNode]);
            // for(int d=0; d<numAgentActions; d++){
            //     if(env.validAgentAction(m, d)){
            //         policy[m][currNode][d] = currUnit->policyOutput->data[(symDir[symID][0]*d + symDir[symID][1] + 4) % 4];
            //     }
            //     else{
            //         policy[m][currNode][d] = -1;
            //     }
                
            //     assert(env.isEndState() || (env.validAgentAction(m, d) == (policy[m][currNode][d] >= 0)));
            // }
        }
    }
    // a trick to regularize values for two-player zero-sum games.
    if(currNode == -1) return;
    values[currNode] = (pred_values[ACTIVE_AGENT] - pred_values[ADVERSARY_AGENT]) / 2;

}

// void MCTSModel::initActivations(int depth){
//     if(pathActivations.size() <= depth){
//         assert(pathActivations.size() == depth);
//         pathActivations.push_back(LSTM::PVUnit(a, &pathActivations[depth-1]));
//         pathActivations[pathActivations.size() - 1].copyParams(&a);
//     }
// }

void MCTSModel::expandPath(){
    int currNode = rootIndex;

    int expandAction;

    int count = 0;
    int currType;
    int maxIndex;
    double maxVal,candVal;
    int i;
    Environment env = rootEnv;

    // if(pathActivations.size() == 0){
    //     pathActivations.push_back(LSTM::PVUnit(a, NULL));
    // }
    // pathActivations[0].copyParams(&a);
    // pathActivations[0].copyAct(&a);
    currUnit->copyParams(a);
    currUnit->copyAct(a);
    nextUnit->copyParams(a);

    for(int i=0; i<2*maxTime; i++){
        times[i] = -1;
    }
    // {
    //     ofstream fout("games.out", ios::app);
    //     fout<<"Path start\n";
    //     fout.close();
    // }
    // ofstream fout("games.out", ios::app);
    // fout<<"Path start\n";
    // fout.close();
    
    while(currNode != -1 && !env.isEndState()){
        // {
        //     ofstream fout("games.out", ios::app);
        //     fout<<currNode<<'\n';
        //     fout.close();
        // }

        // Evaluate environment state using network
        if(count > 0){
            // initActivations(count);
            evaluateEnv(env, -1, nextUnit);
            // copy new hidden state activations into currUnit
            currUnit->copyAct(nextUnit);
        }

        path[count] = currNode;
        env.computeRewards();
        rewards[count] = env.rewards[ACTIVE_AGENT];
        times[count] = env.timer;
        currType = env.actionType;
        maxVal = -1000000;
        maxIndex = -1;
        for(i=0; i<numActions[currType]; i++){
            int numVisits = actionCounts[currNode][i];
            if(env.actionType == 0){
                assert((numVisits != -1) == env.validAgentAction(ACTIVE_AGENT, i));
            }
            else{
                assert((numVisits != -1) == env.validChanceAction(i));
            }
            if(numVisits == -1){ // invalid action
                continue;
            }
            if(currType == 0){
                assert(policy[ACTIVE_AGENT][currNode][i] != -1);
                double Qval;
                int size = 0;
                if(numVisits == 0){
                    if(subtreeSize[currNode] == 0){
                        Qval = 0;
                    }
                    else{
                        Qval = sumScore[currNode] / subtreeSize[currNode];
                    }
                }
                else{
                    Qval = actionSums[currNode][i] / actionCounts[currNode][i];
                    size = actionCounts[currNode][i];
                }
                candVal = Qval + cUCB * policy[ACTIVE_AGENT][currNode][i] * sqrt(subtreeSize[currNode] + 1) / (size + 1);
            }
            if(currType == 1){
                if(numVisits == 0){
                    candVal = (double) rand() / RAND_MAX + 1;
                }
                else{
                    candVal = (double)rand() / RAND_MAX - numVisits;
                }
            }
            if(candVal > maxVal){
                maxVal = candVal;
                maxIndex = i;
            }
        }
        assert(maxIndex != -1);

        Action chosenAction;
        chosenAction.actionType = currType;
        if(currType == 1){
            chosenAction.chanceAction = maxIndex;
            // env.chanceAction(chosenAction);
        }
        else{
            int adversary_action = sampleDist(policy[ADVERSARY_AGENT][currNode], numAgentActions);
            chosenAction.agentActions[ACTIVE_AGENT] = maxIndex;
            chosenAction.agentActions[ADVERSARY_AGENT] = adversary_action;
            // chosenAction = maxIndex*numAgentActions + adversary_action;
            // assert(env.validAgentAction(ACTIVE_AGENT, maxIndex));
            // env.setAgentAction(ACTIVE_AGENT, maxIndex);
            // assert(env.validAgentAction(ADVERSARY_AGENT, adversary_action));
            // env.setAgentAction(ADVERSARY_AGENT, adversary_action);
            // env.agentActions[ACTIVE_AGENT] = maxIndex;
            // env.agentActions[ADVERSARY_AGENT] = adversary_action;
            // env.agentAction();
        }
        env.makeAction(chosenAction);

        currNode = outcomes[currNode][chosenAction.actionID()];
        assert(currNode != -2);

        expandAction = chosenAction.actionID();

        // {
        //     ofstream fout("games.out", ios::app);
        //     fout<<expandAction<<'\n';
        //     fout.close();
        // }

        pathActions[count] = maxIndex;
        count++;
    }
    double newVal;
    if(currNode == -1){
        outcomes[path[count-1]][expandAction] = index;
        initializeNode(env, index);
        // initActivations(count);
        evaluateEnv(env, index, nextUnit);
        
        newVal = values[index];
        
        path[count] = index;
        times[count] = env.timer;
        index++;
        count++;
    }
    else{
        env.computeRewards();
        newVal = env.rewards[ACTIVE_AGENT];
        path[count] = currNode;
        times[count] = env.timer;
        count++;
    }
    double value = newVal;
    for(i=count-1; i>=0; i--){
        subtreeSize[path[i]]++;
        sumScore[path[i]] += value;
        if(i < count-1){
            assert(actionCounts[path[i]][pathActions[i]] != -1);
            actionSums[path[i]][pathActions[i]] += value;
            actionCounts[path[i]][pathActions[i]] ++;
        }
        assert(times[i] >= 0);
        if(i > 0){
            value = rewards[i-1] + value * pow(discountFactor, times[i] - times[i-1]);
        }
    }
}

// int MCTSModel::getAdversaryAction(Environment& env, int currIndex){
//     int advIndex = sampleDist(advProb, numAdversaries);
//     double policy[numAgentActions];
//     if(calculated_adversary_policy[currIndex][advIndex]){
//         for(int d=0; d<numAgentActions; d++){
//             policy[d] = adversary_policy[currIndex][advIndex][d];
//         }
//     }
//     else{
//         int symID = rand() % 8;
//         env.inputSymmetric(adversaries[advIndex], symID, ADVERSARY_AGENT);
//         adversaries[advIndex].pass(PASS_FULL);
//         for(int d=0; d<numAgentActions; d++){
//             policy[d] = adversaries[advIndex].policyOutput[(symDir[symID][0]*d + symDir[symID][1] + 4) % 4];
//             adversary_policy[currIndex][advIndex][d] = policy[d];
//         }
//         calculated_adversary_policy[currIndex][advIndex] = true;
//     }

//     for(int i=0; i<numAgentActions; i++){
//         assert(env.validAgentAction(ADVERSARY_AGENT, i) == (policy[i] >= 0));
//     }
    
//     return sampleDist(policy, numAgentActions);
// }

void MCTSModel::printTree(){
    
//    ofstream fout(outAddress, ios::app);
//    for(int i=0; i<index; i++){
//        fout<<"State "<<i<<'\n';
//        states[i].print();
//        fout<<"Outcomes: ";
//        for(int j=0; j<numActions[states[i].actionType]; j++){
//            fout<<outcomes[i][j];
//        }
//        fout<<'\n';
//        fout<<"Size: "<<size[i]<<'\n';
//        fout<<"Sum score: "<<sumScore[i]<<'\n';
//        fout<<'\n';
//    }
//    fout.close();
}

void MCTSModel::computeActionProbs(){
    int i;
    double sum = 0;
    for(i=0; i<numAgentActions; i++){
        assert(rootEnv.isEndState() || (rootEnv.validAgentAction(ACTIVE_AGENT, i) == (actionCounts[rootIndex][i] != -1)));
        if(actionCounts[rootIndex][i] != -1){
            actionProbs[i] = pow(actionCounts[rootIndex][i], actionTemperature);
            sum += actionProbs[i];
        }
        else{
            actionProbs[i] = -1;
        }
    }
    assert(sum > 0);
    for(i=0; i<numAgentActions; i++){
        if(actionProbs[i] >= 0){
            actionProbs[i] /= sum;
        }
    }
}

int MCTSModel::optActionProbs(){
    int i;
    int maxIndex = 0;
    for(i=1; i<numAgentActions; i++){
        if(actionProbs[i] > actionProbs[maxIndex]){
            maxIndex = i;
        }
    }
    return maxIndex;
}

// int MCTSModel::sampleActionProbs(){
//     int i;
//     double parsum = 0;
//     double randReal = (double)rand() / RAND_MAX;
    
//     int actionIndex = -1;
//     for(i=0; i<numAgentActions; i++){
//         if(actionProbs[i] == -1){
//             continue;
//         }
//         parsum += actionProbs[i];
//         if(randReal <= parsum){
//             actionIndex = i;
//             break;
//         }
//     }
//     return actionIndex;
// }

// int MCTSModel::getRandomChanceAction(Environment* e){
//     int i;
//     int possibleActions[numChanceActions];
//     int numPossibleActions = 0;
//     for(i=0; i<numChanceActions; i++){
//         if(e->validChanceAction(i)){
//             possibleActions[numPossibleActions] = i;
//             numPossibleActions++;
//         }
//     }
//     return possibleActions[rand() % numPossibleActions];
// }

void MCTSModel::simulateAction(Environment& env, Action chosenAction){
    int ID = chosenAction.actionID();
    // {
    //     ofstream fout("games.out", ios::app);
    //     fout<<"Simulating action " << chosenAction.actionID()<<'\n';
    //     fout<<"Start node: "<<rootIndex<<'\n';
    //     fout.close();
    // }
    if(outcomes[rootIndex][ID] == -1){
        outcomes[rootIndex][ID] = index;
        initializeNode(env, index);
        index++;
    }
    rootEnv.makeAction(chosenAction);
    assert(rootEnv == env);
    rootIndex = outcomes[rootIndex][ID];

    nextRoot->copyParams(a);
    evaluateEnv(env, rootIndex, nextRoot);
    a->copyAct(nextRoot);


    // {
    //     ofstream fout("games.out", ios::app);
    //     fout<<"End node: "<<rootIndex<<'\n';
    //     fout.close();
    // }


    if(env.actionType == 0){
        for(int i=0; i<numAgentActions; i++){
            assert((actionCounts[rootIndex][i] != -1) == env.validAgentAction(ACTIVE_AGENT, i));
        }
    }
}