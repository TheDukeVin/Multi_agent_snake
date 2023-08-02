//
//  trainer.cpp
//  trainer
//
//  Created by Kevin Du on 1/18/22.
//

#include "snake.h"


void Trainer::trainGame(int mode){
    // cout<<"TRAIN GAME\n";
    double search_values[maxTime*2];
    double search_policies[maxTime*2][numAgentActions];
    for(int i=0; i<maxTime*2; i++){
        for(int j=0; j<numAgentActions; j++){
            search_policies[i][j] = -1;
        }
    }
    for(int m=0; m<numAgents; m++){
        for(int i=0; i<maxStates; i++){
            for(int j=0; j<numAgentActions; j++){
                models[m].policy[m][i][j] = -1;
            }
        }
    }
    
    roots[0].initialize();
    for(int m=0; m<numAgents; m++){
        // models[m].rootEnv = roots[0];
        models[m].rootIndex = 0;
        models[m].initializeNode(roots[0], 0);
        models[m].index = 1;
    }
    models[TRAIN_ACTIVE].a.copyParam(a);
    models[TRAIN_ADVERSARY].a.copyParam(competitor);
    valueOutput = to_string(roots[0].apple.x * boardy + roots[0].apple.y) + ' ';
    
    // In the multiagent case, chosenAction reflects the actions of ALL players.
    Action chosenAction;

    int r;
    for(r=0; r<maxTime*2; r++){
        // roots[r].log("games.out");
        rootIndices[r] = models[TRAIN_ACTIVE].rootIndex;
        for(int m=0; m<numAgents; m++){
            models[m].rootEnv = roots[r];
        }
        chosenAction.actionType = roots[r].actionType;
        if(roots[r].actionType == 0){
            if(mode == TRAIN_MODE){
                for(int m=0; m<numAgents; m++){
                    for(int j=0; j<numPaths; j++){
                        models[m].expandPath();
                    }
                    models[m].computeActionProbs();
                }
            }
            else{
                for(int m=0; m<numAgents; m++){
                    for(int i=0; i<numAgentActions; i++){
                        models[m].actionProbs[i] = models[m].policy[m][models[m].rootIndex][i];
                    }
                }
            }
            for(int m=0; m<numAgents; m++){
                for(int i=0; i<numAgentActions; i++){
                    assert(roots[r].validAgentAction(m, i) == (models[m].actionProbs[i] != -1));
                }
            }
            for(int j=0; j<numAgentActions; j++){
                search_policies[r][j] = models[TRAIN_ACTIVE].actionProbs[j];
            }
            int active_action = sampleDist(models[TRAIN_ACTIVE].actionProbs, numAgentActions);
            int adversary_action = sampleDist(models[TRAIN_ADVERSARY].actionProbs, numAgentActions);
            
            chosenAction.agentActions[TRAIN_ACTIVE] = active_action;
            chosenAction.agentActions[TRAIN_ADVERSARY] = adversary_action;

            // roots[r+1] = roots[r];
            // roots[r+1].setAgentAction(TRAIN_ACTIVE, active_action);
            // roots[r+1].setAgentAction(TRAIN_ADVERSARY, adversary_action);
            // roots[r+1].agentAction();
        }
        else{
            chosenAction.chanceAction = getRandomChanceAction(&roots[r]);
            // roots[r+1] = roots[r];
            // roots[r+1].chanceAction(chosenAction);
        }
        roots[r+1] = roots[r];
        roots[r+1].makeAction(chosenAction);

        int rIndex = models[TRAIN_ACTIVE].rootIndex;
        if(models[TRAIN_ACTIVE].subtreeSize[rIndex] != 0){
            search_values[r] = models[TRAIN_ACTIVE].sumScore[rIndex] / models[TRAIN_ACTIVE].subtreeSize[rIndex];
        }
        else{
            search_values[r] = 0;
        }

        for(int m=0; m<numAgents; m++){
            models[m].simulateAction(roots[r+1], chosenAction);
        }

        valueOutput += to_string(chosenAction.actionID()) + ' ';
        if(roots[r+1].isEndState()){
            break;
        }
    }
    valueOutput += "\n";

    int numStates = r + 2;
    Data* game = new Data[numStates];

    total_reward = 0;
    for(int i=0; i<numStates; i++){
        roots[i].computeRewards();
        total_reward += roots[i].rewards[TRAIN_ACTIVE];
    }

    roots[numStates-1].computeRewards();
    double value = roots[numStates-1].rewards[TRAIN_ACTIVE];
    for(int i=numStates-1; i>=0; i--){
        game[i] = Data(&roots[i], value);
        if(i > 0){
            roots[i-1].computeRewards();
            value = roots[i-1].rewards[TRAIN_ACTIVE] + value * pow(discountFactor, roots[i].timer - roots[i-1].timer);
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

    // Log true value
    for(int i=0; i<numStates; i++){
        valueOutput += to_string(game[i].expectedValue) + ' ';
    }
    valueOutput += "\n";

    // Log predicted value
    for(int i=0; i<numStates; i++){
        valueOutput += to_string(models[TRAIN_ACTIVE].values[rootIndices[i]]) + ' ';
    }
    valueOutput += "\n";

    // Log search value
    search_values[numStates - 1] = game[numStates - 1].e.rewards[TRAIN_ACTIVE];
    for(int i=0; i<numStates; i++){
        valueOutput += to_string(search_values[i]);
        if(i != numStates-1) valueOutput += ' ';
    }
    valueOutput += "\n";

    // Log predicted policy
    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            valueOutput += to_string(models[TRAIN_ACTIVE].policy[TRAIN_ACTIVE][rootIndices[i]][j]) + ' ';
        }
    }
    valueOutput += "\n";

    // Log search policy
    for(int i=0; i<numStates; i++){
        for(int j=0; j<numAgentActions; j++){
            valueOutput += to_string(search_policies[i][j]) + ' ';
        }
    }
    valueOutput += "\n";
//
//    return &roots[numStates-1];
}

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


void Trainer::passParams(){
    models[TRAIN_ACTIVE].ACTIVE_AGENT = TRAIN_ACTIVE;
    models[TRAIN_ACTIVE].ADVERSARY_AGENT = TRAIN_ADVERSARY;
    models[TRAIN_ADVERSARY].ACTIVE_AGENT = TRAIN_ADVERSARY;
    models[TRAIN_ADVERSARY].ADVERSARY_AGENT = TRAIN_ACTIVE;
    for(int m=0; m<numAgents; m++){
        models[m].actionTemperature = actionTemperature;
        models[m].cUCB = cUCB;
    }
    models[TRAIN_ACTIVE].a = a;
    models[TRAIN_ADVERSARY].a = competitor;
}