/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include "snake.h"

// dq is too big to be defined in the Trainer class, so it is defined outside.

DataQueue dq;
Trainer* trainers[NUM_THREADS];
thread* threads[NUM_THREADS];

Trainer testing_ground;

// tournament[i][j] for i<j
// i is adversary, j is active agent - score of active agent.

double tournament[maxAgentQueue][maxAgentQueue];

unsigned long start_time;

void standardSetup(Agent& net){
    net.commonBranch.initEnvironmentInput(15, 10, 10, 3, 3);
    net.commonBranch.addConvLayer(15, 10, 10, 3, 3);
    net.commonBranch.addPoolLayer(15, 5, 5);
    net.setupCommonBranch();
    net.policyBranch.addFullyConnectedLayer(300);
    net.policyBranch.addFullyConnectedLayer(200);
    net.policyBranch.addOutputLayer(4);
    net.valueBranch.addFullyConnectedLayer(300);
    net.valueBranch.addFullyConnectedLayer(200);
    net.valueBranch.addOutputLayer(1);
    net.setup();
    net.randomize(0.2);
}

void printArray(double* A, int size){
    for(int i=0; i<size; i++){
        cout<<A[i]<<' ';
    }
    cout<<'\n';
}

double getError(Agent& net){
    double error = squ(net.valueOutput - net.valueExpected);
    for(int i=0; i<4; i++){
        if(net.validAction[i]){
            error -= net.policyExpected[i] * log(net.policyOutput[i]);
        }
    }
    return error;
}

void testNet(){
    cout<<"Testing net:\n";
    Agent net;
    standardSetup(net);
    net.randomize(0.5);

    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            net.input->snake[i][j] = rand() % 14 - 1;
        }
    }
    for(int i=0; i<4; i++){
        net.validAction[i] = rand() % 2;
    }
    net.validAction[rand() % 4] = 1;

    net.valueExpected = (double) rand() / RAND_MAX;
    double sum = 0;
    for(int i=0; i<4; i++){
        if(net.validAction[i]){
            net.policyExpected[i] = (double) rand() / RAND_MAX;
            sum += net.policyExpected[i];
        }
    }
    double sum2 = 0;
    for(int i=0; i<4; i++){
        if(net.validAction[i]){
            net.policyExpected[i] /= sum;
            sum2 += net.policyExpected[i];
        }
    }
    
    net.pass(PASS_FULL);
    double base = getError(net);

    net.backProp(PASS_FULL);
    double ep = 0.000001;
    
    for(int l=0; l<net.numLayers; l++){
        for(int i=0; i<net.layers[l]->numParams; i++){
            net.layers[l]->params[i] += ep;
            net.pass(PASS_FULL);
            net.layers[l]->params[i] -= ep;
            double new_error = getError(net);
            //cout<< ((new_error - base) / ep) << ' ' << net.layers[l]->Dparams[i]<<'\n';
            assert( abs((new_error - base) / ep - net.layers[l]->Dparams[i]) < 0.01);
        }
    }
}

void runThread(Trainer* t){
    t->trainTree(TRAIN_MODE);
}

void trainCycle(){
    cout<<"Beginning training: "<<time(NULL)<<'\n';
    Agent a;
    standardSetup(a);

    // a.readNet("nets/Game400.out");

    for(int i=0; i<NUM_THREADS; i++){
        trainers[i] = new Trainer();
        standardSetup(trainers[i]->a);
        trainers[i]->numAdversaries = 1;
        standardSetup(trainers[i]->adversaries[0]);
        //trainers[i]->adversaries[0].readNet("nets/Game1600.out");
        trainers[i]->advProb[0] = 1;
    }

    int tournament_size = 1;
    standardSetup(testing_ground.a);
    testing_ground.numAdversaries = 1;
    standardSetup(testing_ground.adversaries[0]);
    testing_ground.advProb[0] = 1;
    //cout<<"Reading net:\n";
    //t.a.readNet("snakeConv.in");

    const int storePeriod = 1000;
    
    dq.index = 0;
    dq.currSize = 500;
    dq.momentum = 0.7;
    dq.learnRate = 0.001;
    double explorationConstant = 0.5;
    //t.actionTemperature = 2;
    
    //cout<<"Reading games\n";
    //vector<int> scores = dq.readGames(); // read games from games.in file.
    //cout<<"Finished reading " << dq.index << " games\n";
    vector<int> scores;
    
    double sum = 0;
    int completions = 0;
    double completionTime = 0;
    
    string summaryLog = "summary.out";
    string controlLog = "control.out";
    string valueLog = "details.out";
    string scoreLog = "scores.out";
    string gameLog = "games.out";
    ofstream hold2(summaryLog);
    hold2.close();
    ofstream hold3(valueLog);
    hold3.close();
    ofstream hold4(scoreLog);
    hold4.close();
    ofstream hold5(controlLog);
    hold5.close();
    ofstream hold6(gameLog);
    hold6.close();
    
    for(int i=1; i<=numGames; ){

        for(int j=0; j<NUM_THREADS; j++){
            trainers[j]->a.copyParam(a);
            trainers[j]->explorationConstant = explorationConstant;
            threads[j] = new thread(runThread, trainers[j]);
        }
        for(int j=0; j<NUM_THREADS; j++){
            threads[j]->join();
            dq.enqueue(trainers[j]->output_game, trainers[j]->output_gameLength);
        }

        for(int j=0; j<NUM_THREADS; j++){
            // Log games.
            ofstream gameOut(gameLog, ios::app);
            gameOut<<"Game "<<i<<":\n";
            gameOut.close();
            for(int t=0; t<trainers[j]->output_gameLength; t++){
                trainers[j]->output_game[t].e.log(gameLog);
            }

            ofstream valueOut(valueLog, ios::app);
            valueOut<<"Game "<<i<<' '<<time(NULL)<<'\n';
            valueOut<<trainers[j]->valueOutput;
            valueOut.close();

            // get cumulative rewards
            double score = trainers[j]->total_reward;
            Environment* result = &(trainers[j]->output_game[trainers[j]->output_gameLength-1].e);
            // double score = result->snakes[0].size - result->snakes[1].size;
            sum += score;
            
            ofstream summaryOut(summaryLog, ios::app);
            summaryOut<<i<<':'<<score<<' '<<result->timer<<' '<<(time(NULL) - start_time)<<'\n';
            summaryOut.close();

            scores.push_back(score);
            ofstream scoreOut(scoreLog);
            for(int s=0; s<scores.size(); s++){
                if(s > 0){
                    scoreOut<<',';
                }
                scoreOut<<scores[s];
            }
            scoreOut<<'\n';
            scoreOut.close();


            // add an adversary
            if(i>0 && i%evalPeriod == 0){
                ofstream controlOut(controlLog, ios::app);
                controlOut << "Game "<<i<<'\n';

                for(int t=0; t<tournament_size; t++){
                    testing_ground.a.copyParam(a);
                    testing_ground.adversaries[0].copyParam(trainers[0]->adversaries[t]);
                    double sum = 0;
                    controlOut<<"Testing agent " << t << " and agent "<<tournament_size<<'\n';
                    int symFactor = 1;
                    for(int it=0; it<numEvalGames; it++){
                        if(it == numEvalGames / 2){
                            // swap two agents, if environment is not symmetric.
                            testing_ground.a.copyParam(trainers[0]->adversaries[t]);
                            testing_ground.adversaries[0].copyParam(a);
                            symFactor = -1;
                        }
                        testing_ground.trainTree(TEST_MODE);
                        controlOut << testing_ground.total_reward * symFactor << ' ';
                        sum += testing_ground.total_reward / scoreNorm * symFactor;
                    }
                    controlOut << '\n';
                    tournament[t][tournament_size] = sum / numEvalGames;
                }
                tournament_size ++;

                Nash NE(tournament_size, tournament_size);
                for(int i_=0; i_<tournament_size; i_++){
                    for(int j_=i_+1; j_<tournament_size; j_++){
                        NE.A[i_][j_] = tournament[i_][j_];
                        NE.A[j_][i_] = -tournament[i_][j_];
                    }
                    NE.A[i_][i_] = 0;
                }
                NE.find_equilibrium(1e+08, 1e-05);

                controlOut << "Tournament matrix: \n";
                for(int k=0; k<tournament_size; k++){
                    for(int l=0; l<tournament_size; l++){
                        controlOut << NE.A[k][l] << ' ';
                    }
                    controlOut << '\n';
                }

                controlOut << "Equilibrium: \n";
                for(int k=0; k<tournament_size; k++){
                    controlOut << NE.p1[k] << ' ';
                }
                controlOut<<'\n';
                controlOut << "EXPLOITABILITY: " << NE.exploitabilty() << '\n';
                controlOut.close();

                for(int t=0; t<NUM_THREADS; t++){
                    int numAdv = trainers[t]->numAdversaries;
                    standardSetup(trainers[t]->adversaries[numAdv]);
                    trainers[t]->adversaries[numAdv].copyParam(a);
                    trainers[t]->numAdversaries++;
                    for(int k=0; k<trainers[t]->numAdversaries; k++){
                        trainers[t]->advProb[k] = NE.p1[k];
                    }
                }
            }
            
            // if(i>0 && i%evalPeriod == 0){
            //     ofstream controlOut(controlLog, ios::app);
            //     controlOut<<"\nAVERAGE: "<<(sum / evalPeriod)<<" in iteration "<<i<<'\n';
            //     controlOut<<"Completions: "<<((double) completions / evalPeriod)<<'\n';
            //     if(completions > 0){
            //         controlOut<<"Average completion time: "<<(completionTime / completions)<<'\n';
            //     }
            //     controlOut<<" TIMESTAMP: "<<(time(NULL) - start_time)<<'\n';

            //     if(sum / evalPeriod > 80){
            //         dq.currSize = max(2000, dq.currSize);
            //         explorationConstant = min(0.4, explorationConstant);
            //         controlOut<<"Queue set to "<<dq.currSize<<'\n';
            //         controlOut<<"Exploration constant set to "<<explorationConstant<<'\n';
            //     }
            //     if(sum / evalPeriod > 95){
            //         dq.currSize = max(10000, dq.currSize);
            //         explorationConstant = min(0.3, explorationConstant);
            //         controlOut<<"Queue set to "<<dq.currSize<<'\n';
            //         controlOut<<"Exploration constant set to "<<explorationConstant<<'\n';
            //     }
            //     controlOut.close();

            //     sum = 0;
            //     completions = 0;
            //     completionTime = 0;
            // }
            if(i % storePeriod == 0){
                a.save("nets/Game" + to_string(i) + ".out");
            }
            
            dq.trainAgent(a);
            i++;
        }
    }
    
}

vector<Environment> nextStates(Environment e){
    vector<Environment> states;
    if(e.actionType == 0){
        for(int i=0; i<numAgentActions; i++){
            if(!e.validAgentAction(0, i)) continue;
            for(int j=0; j<numAgentActions; j++){
                if(!e.validAgentAction(1, j)) continue;
                Environment newEnv = e;
                newEnv.agentActions[0] = i;
                newEnv.agentActions[1] = j;
                newEnv.agentAction();
                states.push_back(newEnv);
            }
        }
    }
    else{
        for(int i=0; i<numChanceActions; i++){
            if(!e.validChanceAction(i)) continue;
            Environment newEnv = e;
            newEnv.chanceAction(i);
            states.push_back(newEnv);
        }
    }
    
    return states;
}

void computeEnv(){
    Environment env;
    env.initialize();
    list<Environment> queue;
    queue.push_back(env);
    unordered_map<Environment, bool, EnvHash> states;
    vector<Environment> allStates;

    while(queue.size() != 0){
        Environment currEnv = queue.front();
        //currEnv.log();
        queue.pop_front();
        if(currEnv.isEndState()){
            continue;
        }
        for(auto nextEnv : nextStates(currEnv)){
            if(states.find(nextEnv) == states.end()){
                states[nextEnv] = true;
                queue.push_back(nextEnv);
                allStates.push_back(nextEnv);
            }
        }
    }
    cout<<"Number of states: "<<states.size()<<'\n';
/*
    for(int i=0; i<10; i++){
        allStates[rand() % allStates.size()].log();
    }*/
}

int main()
{
    srand((unsigned)time(NULL));
    start_time = time(NULL);
    
    /*
    for(int i=0; i<1; i++){
        testNet();
    }*/
    
    trainCycle();
    
    //evaluate();

    //computeEnv();
    
    return 0;
    
}



