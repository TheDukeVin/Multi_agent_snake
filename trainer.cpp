//
//  trainer.cpp
//  trainer
//
//  Created by Kevin Du on 1/18/22.
//

#include "snake.h"


void Trainer::initializeNode(Environment& env, int currNode){
    if(outcomes[currNode] != NULL){
        delete outcomes[currNode];
    }
    //int numOutcomes = numActions[env.actionType];
    if(env.actionType == 0){
        outcomes[currNode] = new int[numAgentActions * numAgentActions];
        for(int i=0; i<numAgentActions; i++){
            for(int j=0; j<numAgentActions; j++){
                if(env.validAgentAction(ACTIVE_AGENT, i) && env.validAgentAction(ADVERSARY_AGENT, j)){
                    outcomes[currNode][i*numAgentActions + j] = -1;
                }
                else{
                    outcomes[currNode][i*numAgentActions + j] = -2;
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
    
    // Evaluate the network at the current node.
    
    
    int symID = rand()%8;
    env.inputSymmetric(a, symID, ACTIVE_AGENT);
    if(env.actionType == 0){
        a.pass(PASS_FULL);
    }
    else{
        a.pass(PASS_VALUE);
    }
    values[currNode] = a.valueOutput;
    if(env.actionType == 0){
        for(int d=0; d<numAgentActions; d++){
            if(env.validAgentAction(ACTIVE_AGENT, d)){
                policy[currNode][d] = a.policyOutput[(symDir[symID][0]*d + symDir[symID][1] + 4) % 4];
            }
            else{
                policy[currNode][d] = -1;
            }
            
            assert(env.isEndState() || (env.validAgentAction(ACTIVE_AGENT, d) == (policy[currNode][d] >= 0)));
        }
    }
    for(int i=0; i<numAdversaries; i++){
        calculated_adversary_policy[currNode][i] = false;
    }
}

void Trainer::trainTree(int mode){
    double search_values[maxTime*2];
    double search_policies[maxTime*2][numAgentActions];
    for(int i=0; i<maxTime*2; i++){
        for(int j=0; j<numAgentActions; j++){
            search_policies[i][j] = -1;
        }
    }
    for(int i=0; i<maxStates; i++){
        for(int j=0; j<numAgentActions; j++){
            policy[i][j] = -1;
        }
    }
    
    roots[0].initialize();
    rootIndex = 0;
    valueOutput = to_string(roots[0].apple.x * boardy + roots[0].apple.y) + ' ';
    initializeNode(roots[0], 0);
    index = 1;
    
    // In the multiagent case, chosenAction reflects the actions of ALL players.
    int chosenAction;

    for(rootState=0; rootState<maxTime*2; rootState++){
        // roots[rootState].log();
        rootIndices[rootState] = rootIndex;
        if(roots[rootState].actionType == 0){
            if(mode == TRAIN_MODE){
                for(int j=0; j<numPaths; j++){
                    expandPath();
                }
                computeActionProbs();
            }
            else{
                for(int i=0; i<numAgentActions; i++){
                    actionProbs[i] = policy[rootIndex][i];
                }
            }
            for(int i=0; i<numAgentActions; i++){
                assert(roots[rootState].validAgentAction(ACTIVE_AGENT, i) == (actionProbs[i] != -1));
            }
            for(int j=0; j<numAgentActions; j++){
                search_policies[rootState][j] = actionProbs[j];
            }
            int active_action = sampleDist(actionProbs, numAgentActions);
            int adversary_action = getAdversaryAction(roots[rootState], rootIndex);
            chosenAction = active_action * numAgentActions + adversary_action;

            roots[rootState+1] = roots[rootState];
            roots[rootState+1].agentActions[ACTIVE_AGENT] = active_action;
            assert(roots[rootState+1].validAgentAction(ACTIVE_AGENT, active_action));
            roots[rootState+1].agentActions[ADVERSARY_AGENT] = adversary_action;
            assert(roots[rootState+1].validAgentAction(ADVERSARY_AGENT, adversary_action));
            roots[rootState+1].agentAction();
        }
        else{
            chosenAction = getRandomChanceAction(&roots[rootState]);
            roots[rootState+1] = roots[rootState];
            roots[rootState+1].chanceAction(chosenAction);
        }
        if(subtreeSize[rootIndex] != 0){
            search_values[rootState] = sumScore[rootIndex] / subtreeSize[rootIndex];
        }
        else{
            search_values[rootState] = 0;
        }
        if(outcomes[rootIndex][chosenAction] == -1){
            outcomes[rootIndex][chosenAction] = index;
            initializeNode(roots[rootState+1], index);
            index++;
        }
        rootIndex = outcomes[rootIndex][chosenAction];
        valueOutput += to_string(chosenAction) + ' ';
        if(roots[rootState+1].isEndState()){
            break;
        }
    }
    valueOutput += "\n";

    int numStates = rootState + 2;
    Data* game = new Data[numStates];

    total_reward = 0;
    for(int i=0; i<numStates; i++){
        roots[i].computeRewards();
        total_reward += roots[i].rewards[ACTIVE_AGENT];
    }

    roots[numStates-1].computeRewards();
    double value = roots[numStates-1].rewards[ACTIVE_AGENT];
    for(int i=numStates-1; i>=0; i--){
        game[i] = Data(&roots[i], value);
        if(i > 0){
            roots[i-1].computeRewards();
            value = roots[i-1].rewards[ACTIVE_AGENT] + value * pow(discountFactor, roots[i].timer - roots[i-1].timer);
        }
    }
    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            game[i].expectedPolicy[j] = search_policies[i][j];
        }
    }
    //dq->enqueue(game, numStates);
    output_gameLength = numStates;
    output_game = game;

    for(int i=0; i<numStates; i++){
        valueOutput += to_string(game[i].expectedValue) + ' ';
    }
    valueOutput += "\n";

    for(int i=0; i<numStates; i++){
        valueOutput += to_string(values[rootIndices[i]]) + ' ';
    }
    valueOutput += "\n";

    search_values[numStates - 1] = game[numStates - 1].e.rewards[ACTIVE_AGENT];
    for(int i=0; i<numStates; i++){
        valueOutput += to_string(search_values[i]);
        if(i != numStates-1) valueOutput += ' ';
    }
    valueOutput += "\n";

    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            valueOutput += to_string(policy[rootIndices[i]][j]) + ' ';
        }
    }
    valueOutput += "\n";

    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            valueOutput += to_string(search_policies[i][j]) + ' ';
        }
    }
    valueOutput += "\n";
//
//    return &roots[numStates-1];
}

void Trainer::expandPath(){
    int currNode = rootIndex;

    int expandAction;

    int count = 0;
    int currType;
    int maxIndex;
    double maxVal,candVal;
    int i;
    Environment env = roots[rootState];

    for(int i=0; i<2*maxTime; i++){
        times[i] = -1;
    }
    
    while(currNode != -1 && !env.isEndState()){
        path[count] = currNode;
        env.computeRewards();
        rewards[count] = env.rewards[ACTIVE_AGENT];
        times[count] = env.timer;
        currType = env.actionType;
        maxVal = -1000000;
        maxIndex = -1;
        for(i=0; i<numActions[currType]; i++){
            int numVisits = actionCounts[currNode][i];
            if(numVisits == -1){ // invalid action
                continue;
            }
            if(currType == 0){
                assert(policy[currNode][i] != -1);
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
                candVal = Qval + explorationConstant * policy[currNode][i] * sqrt(subtreeSize[currNode] + 1) / (size + 1);
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

        int chosenAction;
        if(currType == 1){
            chosenAction = maxIndex;
            env.chanceAction(chosenAction);
        }
        else{
            int adversary_action = getAdversaryAction(env, currNode);
            chosenAction = maxIndex*numAgentActions + adversary_action;
            env.agentActions[ACTIVE_AGENT] = maxIndex;
            env.agentActions[ADVERSARY_AGENT] = adversary_action;
            env.agentAction();
        }
        currNode = outcomes[currNode][chosenAction];
        assert(currNode != -2);

        expandAction = chosenAction;
        pathActions[count] = maxIndex;
        count++;
    }
    double newVal;
    if(currNode == -1){
        outcomes[path[count-1]][expandAction] = index;
        initializeNode(env, index);
        
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

int Trainer::getAdversaryAction(Environment& env, int currIndex){
    int advIndex = sampleDist(advProb, numAdversaries);
    double policy[numAgentActions];
    if(calculated_adversary_policy[currIndex][advIndex]){
        for(int d=0; d<numAgentActions; d++){
            policy[d] = adversary_policy[currIndex][advIndex][d];
        }
    }
    else{
        int symID = rand() % 8;
        env.inputSymmetric(adversaries[advIndex], symID, ADVERSARY_AGENT);
        adversaries[advIndex].pass(PASS_FULL);
        for(int d=0; d<numAgentActions; d++){
            policy[d] = adversaries[advIndex].policyOutput[(symDir[symID][0]*d + symDir[symID][1] + 4) % 4];
            adversary_policy[currIndex][advIndex][d] = policy[d];
        }
        calculated_adversary_policy[currIndex][advIndex] = true;
    }

    for(int i=0; i<numAgentActions; i++){
        assert(env.validAgentAction(ADVERSARY_AGENT, i) == (policy[i] >= 0));
    }
    
    return sampleDist(policy, numAgentActions);
}

void Trainer::printTree(){
    
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

void Trainer::computeActionProbs(){
    int i;
    double sum = 0;
    for(i=0; i<numAgentActions; i++){
        assert(roots[rootState].isEndState() || (roots[rootState].validAgentAction(ACTIVE_AGENT, i) == (actionCounts[rootIndex][i] != -1)));
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

int Trainer::optActionProbs(){
    int i;
    int maxIndex = 0;
    for(i=1; i<numAgentActions; i++){
        if(actionProbs[i] > actionProbs[maxIndex]){
            maxIndex = i;
        }
    }
    return maxIndex;
}

// int Trainer::sampleActionProbs(){
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

int Trainer::getRandomChanceAction(Environment* e){
    int i;
    int possibleActions[numChanceActions];
    int numPossibleActions = 0;
    for(i=0; i<numChanceActions; i++){
        if(e->validChanceAction(i)){
            possibleActions[numPossibleActions] = i;
            numPossibleActions++;
        }
    }
    return possibleActions[rand() % numPossibleActions];
}

