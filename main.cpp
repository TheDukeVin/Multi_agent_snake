/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include "snake.h"
// #include "LSTM/test.h"

using namespace std;



// dq is too big to be defined in the Trainer class, so it is defined outside.

DataQueue* dq;
Trainer* trainers[NUM_THREADS];
thread* threads[NUM_THREADS];

Trainer* testing_ground;

// tournament[i][j] for i<j
// i is adversary, j is active agent - score of active agent.

double tournament[maxAgentQueue][maxAgentQueue];
double nashDist[maxAgentQueue];
LSTM::PVUnit* opponents[maxAgentQueue];

unsigned long start_time;

// void standardSetup(Agent& net){
//     net.commonBranch.initEnvironmentInput(10, 10, 10, 3, 3);
//     net.commonBranch.addConvLayer(15, 10, 10, 3, 3);
//     net.commonBranch.addPoolLayer(15, 5, 5);
//     net.setupCommonBranch();
//     net.policyBranch.addFullyConnectedLayer(200);
//     net.policyBranch.addFullyConnectedLayer(100);
//     net.policyBranch.addOutputLayer(4);
//     net.valueBranch.addFullyConnectedLayer(200);
//     net.valueBranch.addFullyConnectedLayer(100);
//     net.valueBranch.addOutputLayer(1);
//     net.setup();
//     net.randomize(0.2);
// }

void printArray(double* A, int size){
    for(int i=0; i<size; i++){
        cout<<A[i]<<' ';
    }
    cout<<'\n';
}

// double getError(Agent& net){
//     double error = squ(net.valueOutput - net.valueExpected);
//     for(int i=0; i<4; i++){
//         if(net.validAction[i]){
//             error -= net.policyExpected[i] * log(net.policyOutput[i]);
//         }
//     }
//     return error;
// }

// void testNet(){
//     cout<<"Testing net:\n";
//     Agent net;
//     standardSetup(net);
//     net.randomize(0.5);

//     for(int i=0; i<boardx; i++){
//         for(int j=0; j<boardy; j++){
//             net.input->snake[i][j] = rand() % 14 - 1;
//         }
//     }
//     for(int i=0; i<4; i++){
//         net.validAction[i] = rand() % 2;
//     }
//     net.validAction[rand() % 4] = 1;

//     net.valueExpected = (double) rand() / RAND_MAX;
//     double sum = 0;
//     for(int i=0; i<4; i++){
//         if(net.validAction[i]){
//             net.policyExpected[i] = (double) rand() / RAND_MAX;
//             sum += net.policyExpected[i];
//         }
//     }
//     double sum2 = 0;
//     for(int i=0; i<4; i++){
//         if(net.validAction[i]){
//             net.policyExpected[i] /= sum;
//             sum2 += net.policyExpected[i];
//         }
//     }
    
//     net.pass(PASS_FULL);
//     double base = getError(net);

//     net.backProp(PASS_FULL);
//     double ep = 0.000001;
    
//     for(int l=0; l<net.numLayers; l++){
//         for(int i=0; i<net.layers[l]->numParams; i++){
//             net.layers[l]->params[i] += ep;
//             net.pass(PASS_FULL);
//             net.layers[l]->params[i] -= ep;
//             double new_error = getError(net);
//             //cout<< ((new_error - base) / ep) << ' ' << net.layers[l]->Dparams[i]<<'\n';
//             assert( abs((new_error - base) / ep - net.layers[l]->Dparams[i]) < 0.01);
//         }
//     }
// }

OutputGameDetails gameStore[NUM_THREADS][numRolloutsPerThread];

void runThread(int trainerIndex){
    for(int i=0; i<numRolloutsPerThread; i++){
        trainers[trainerIndex]->trainGame(TRAIN_MODE);
        gameStore[trainerIndex][i] = OutputGameDetails(trainers[trainerIndex]->output_game, trainers[trainerIndex]->total_reward);
    }
}

void evaluateAgent(int tournament_size, LSTM::PVUnit* a, string controlLog){
    ofstream controlOut(controlLog, ios::app);

    for(int t=0; t<tournament_size; t++){
        testing_ground->a->copyParams(a);
        testing_ground->competitor->copyParams(opponents[t]);
        testing_ground->passParams();
        double sum = 0;
        controlOut<<"Testing agent " << t << " and agent "<<tournament_size<<'\n';
        int symFactor = 1;
        for(int it=0; it<numEvalGames; it++){
            if(it == numEvalGames / 2){
                // swap two agents, if environment is not symmetric.
                testing_ground->a->copyParams(opponents[t]);
                testing_ground->competitor->copyParams(a);
                symFactor = -1;
            }
            testing_ground->trainGame(TEST_MODE);
            controlOut << testing_ground->total_reward * symFactor << ' ';
            sum += testing_ground->total_reward / scoreNorm * symFactor;
        }
        controlOut << '\n';
        tournament[t][tournament_size] = sum / numEvalGames;
    }

    
}

void computeOracle(int tournament_size, string controlLog){
    ofstream controlOut(controlLog, ios::app);
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

    for(int k=0; k<tournament_size; k++){
        nashDist[k] = NE.p1[k];
    }
    controlOut.close();
}

void trainCycle(){
    cout<<"Beginning training: "<<time(NULL)<<'\n';

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(10, 10, 13));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 15), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 15));
    // structure.initPV();
    // structure.policyBranch->addLSTM(200);
    // structure.policyBranch->addLSTM(100);
    // structure.policyBranch->addOutput(4);
    // structure.valueBranch->addLSTM(200);
    // structure.valueBranch->addLSTM(100);
    // structure.valueBranch->addOutput(1);
    // structure.randomize(0.1);

    LSTM::PVUnit structure;
    structure.commonBranch = new LSTM::Model(LSTM::Shape(10, 10, 13));
    structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    structure.commonBranch->addPool(LSTM::Shape(5, 5, 10));
    structure.initPV();
    structure.policyBranch->addLSTM(50);
    structure.policyBranch->addLSTM(25);
    structure.policyBranch->addOutput(4);
    structure.valueBranch->addLSTM(50);
    structure.valueBranch->addLSTM(25);
    structure.valueBranch->addOutput(1);
    structure.randomize(0.1);

    cout << "Creating agent structure\n";

    LSTM::PVUnit a(&structure, NULL);
    cout << "Copying\n";
    a.copyParams(&structure);

    cout << "Creating trainer structures\n";

    for(int i=0; i<NUM_THREADS; i++){
        trainers[i] = new Trainer(&structure);
    }

    cout << "Creating data queue\n";

    dq = new DataQueue(&structure);

    cout << "Creating opponents\n";

    int tournament_size = 1;
    for(int i=0; i<maxAgentQueue; i++){
        opponents[i] = new LSTM::PVUnit(&structure, NULL);
    }
    nashDist[0] = 1;

    cout << "Creating testing ground\n";

    testing_ground = new Trainer(&structure);

    cout << "Initializing training process\n";

    //cout<<"Reading net:\n";
    //t.a.readNet("snakeConv.in");

    const int storePeriod = 1000;
    
    dq->index = 0;
    dq->currSize = 2000;
    dq->momentum = 0.7;
    dq->learnRate = 0.001;
    double explorationConstant = 0.5;
    
    //cout<<"Reading games\n";
    //vector<int> scores = dq->readGames(); // read games from games.in file.
    //cout<<"Finished reading " << dq->index << " games\n";
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
    {
        ofstream controlOut(controlLog, ios::app);
        controlOut << "Training control: "<<time(NULL)<<'\n';
        controlOut.close();
    }

    cout << "Running games\n";
    
    for(int i=1; i<=numGames; ){
        // Generate training rollouts
        for(int j=0; j<NUM_THREADS; j++){
            trainers[j]->cUCB = explorationConstant;

            trainers[j]->a->copyParams(&a);
            int advIndex = sampleDist(nashDist, tournament_size);
            trainers[j]->competitor->copyParams(opponents[advIndex]);
            trainers[j]->passParams();

            threads[j] = new thread(runThread, j);
        }
        for(int j=0; j<NUM_THREADS; j++){
            threads[j]->join();
        }

        for(int j=0; j<NUM_THREADS*numRolloutsPerThread; j++){
            int trainerIndex = j / numRolloutsPerThread;
            int gameIndex = j % numRolloutsPerThread;
            vector<Data> game = gameStore[trainerIndex][gameIndex].game;
            double total_reward = gameStore[trainerIndex][gameIndex].total_reward;

            dq->enqueue(game);
            // Log games.
            ofstream gameOut(gameLog, ios::app);
            gameOut<<"Game "<<i<<":\n";
            gameOut.close();
            for(int t=0; t<game.size(); t++){
                game[t].e.log(gameLog);
            }

            // ofstream valueOut(valueLog, ios::app);
            // valueOut<<"Game "<<i<<' '<<time(NULL)<<'\n';
            // valueOut<<trainers[j]->valueOutput;
            // valueOut.close();

            // get cumulative rewards
            double score = total_reward;
            Environment* result = &(game[game.size()-1].e);
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
                evaluateAgent(tournament_size, &a, controlLog);
                opponents[tournament_size]->copyParams(&a);
                tournament_size ++;
                computeOracle(tournament_size, controlLog);
            }
            
            if(i>0 && i%trainPeriod == 0){
                cout << "Training\n";
                dq->trainAgent(a, "training" + to_string(i) + ".out");
                cout << "Done training\n";
            }
            i++;

            // Schedule
            if(i > 5000){
                dq->currSize = max(10000, dq->currSize);
                ofstream controlOut(controlLog, ios::app);
                controlOut << "Game queue size set to "<<dq->currSize<<'\n';
                controlOut.close();
            }
        }
    }
}













// Tabular methods

//vector<Environment> nextStates(Environment e){
    // vector<Environment> states;
    // if(e.actionType == 0){
    //     for(int i=0; i<numAgentActions; i++){
    //         if(!e.validAgentAction(0, i)) continue;
    //         for(int j=0; j<numAgentActions; j++){
    //             if(!e.validAgentAction(1, j)) continue;
    //             Environment newEnv = e;
    //             newEnv.agentActions[0] = i;
    //             newEnv.agentActions[1] = j;
    //             newEnv.agentAction();
    //             states.push_back(newEnv);
    //         }
    //     }
    // }
    // else{
    //     for(int i=0; i<numChanceActions; i++){
    //         if(!e.validChanceAction(i)) continue;
    //         Environment newEnv = e;
    //         newEnv.chanceAction(i);
    //         states.push_back(newEnv);
    //     }
    // }
    
    // return states;
//}

// void computeEnv(){
//     Environment env;
//     env.initialize();
//     list<Environment> queue;
//     queue.push_back(env);
//     unordered_map<Environment, bool, EnvHash> states;
//     vector<Environment> allStates;

//     while(queue.size() != 0){
//         Environment currEnv = queue.front();
//         //currEnv.log();
//         queue.pop_front();
//         if(currEnv.isEndState()){
//             continue;
//         }
//         for(auto nextEnv : nextStates(currEnv)){
//             if(states.find(nextEnv) == states.end()){
//                 states[nextEnv] = true;
//                 queue.push_back(nextEnv);
//                 allStates.push_back(nextEnv);
//             }
//         }
//     }
//     cout<<"Number of states: "<<states.size()<<'\n';
// /*
//     for(int i=0; i<10; i++){
//         allStates[rand() % allStates.size()].log();
//     }*/
// }











// K-means Clustering:

// int numActivations = 100;

// void get_embedding(){
//     dq.currSize = 10000;
//     dq.readGames("details.out");
//     vector<Environment> states;
//     for(int i=0; i<dq.currSize; i++){
//         for(int j=0; j<dq.gameLengths[i]; j++){
//             states.push_back(dq.queue[i][j].e);
//         }
//     }
//     cout<<states.size()<<'\n';
//     Agent net;
//     standardSetup(net);
//     net.readNet("nets/Game10000.out");
//     ofstream fout ("embed.out");
//     Branch policy_branch = net.policyBranch;
//     Layer* lastLayer = policy_branch.layers[policy_branch.numLayers-1];
//     for(auto s : states){
//         s.inputSymmetric(net, 0, 0);
//         net.pass(PASS_FULL);
//         for(int i=0; i<numActivations; i++){
//             fout<<lastLayer->inputs[i]<<' ';
//         }
//         fout<<'\n';
//     }
// }

// void cluster(){
//     dq.currSize = 10000;
//     dq.readGames("details.out");
//     vector<Environment> states;
//     for(int i=0; i<dq.currSize; i++){
//         for(int j=0; j<dq.gameLengths[i]; j++){
//             states.push_back(dq.queue[i][j].e);
//         }
//     }
//     int N = states.size();
//     cout<<N<<'\n';
//     ifstream fin ("embed.in");
//     double data[N*numActivations];
//     for(int i=0; i<N; i++){
//         for(int j=0; j<numActivations; j++){
//             fin >> data[i*numActivations + j];
//         }
//     }
//     cout<<"Read data\n";
//     Kmeans km(data, numActivations, N);
//     km.cluster(100, 100, 0.001);
// }

// void testKmeans(){
//     double data[8] = {
//         1, 1,
//         2, 2,
//         1, 6,
//         3, 6
//     };
//     Kmeans km(data, 2, 4);
//     km.cluster(2, 10, 0.001);
//     for(int i=0; i<2; i++){
//         for(int j=0; j<2; j++){
//             cout<<km.centers[i][j]<<' ';
//         }
//         cout<<'\n';
//     }
// }

int main()
{
    srand((unsigned)time(NULL));
    start_time = time(NULL);

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(10, 10, 13));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 15), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 15));
    // structure.initPV();
    // structure.policyBranch->addLSTM(100);
    // structure.policyBranch->addLSTM(50);
    // structure.policyBranch->addOutput(4);
    // structure.valueBranch->addLSTM(100);
    // structure.valueBranch->addLSTM(50);
    // structure.valueBranch->addOutput(1);
    // structure.randomize(0.1);

    // cout << "Initializing trainer...\n";
    // Trainer t(structure);
    // t.trainGame(TRAIN_MODE);
    
    /*
    for(int i=0; i<1; i++){
        testNet();
    }*/
    
    trainCycle();
    
    //evaluate();

    //computeEnv();

    // get_embedding();

    // testKmeans();

    // cluster();

    size_t end_time = time(NULL);
    cout<< "TIME: "<<(end_time - start_time) << '\n';
    
    return 0;
    
}



