
#include "PG.h"

void PG::rollOut(bool printGame){
    Environment env;

    double policy[TIME_HORIZON][NUM_ACTIONS];
    Environment states[TIME_HORIZON];
    int actions[TIME_HORIZON][NUM_ACTIONS];

    int rollOutLength = TIME_HORIZON;
    rollOutRewardSum = 0;
    for(int t=0; t<TIME_HORIZON; t++){
        states[t] = env;
        env.inputObservations(&seq.inputs[t]);
        seq.forwardPassUnit(t);
        if(MODE == SINGLE_ACT){
            computeSoftmax(seq.outputs[t].data, policy[t], env.validActions());
            int action = sampleDist(policy[t], NUM_ACTIONS);
            for(int i=0; i<NUM_ACTIONS; i++){
                actions[t][i] = 0;
            }
            actions[t][action] = 1;
            reward[t] = env.makeAction(action);
        }
        else if(MODE == MULT_ACT){
            // computeSigmoid(seq.outputs[t].data, policy[t], env.validActions());
            // for(int i=0; i<NUM_ACTIONS; i++){
            //     actions[t][i] = -1;
            // }
            // for(auto a : env.validActions()){
            //     double prob[2] = {1-policy[t][a], policy[t][a]};
            //     actions[t][a] = sampleDist(prob, 2);
            // }
            // reward[t] = env.makeAction(actions[t]);
            // // CODE SPECIFIC FOR POKER:
            // for(auto a : env.validActions()){
            //     actions[t][a] = -1;
            // }
        }
        
        if(printGame){ 
            ofstream fout(fileOut, ios::app);
            fout << states[t].toString() << '\n';
            fout << "Policy:\n";
            for(int i=0; i<NUM_ACTIONS; i++){
                fout<<policy[t][i]<<' ';
            }
            fout<<'\n';
            fout.close();
            // CODE SPECIFIC FOR POKER:
            // ofstream fout(fileOut, ios::app);
            // fout << states[t].toString() << '\n';
            // fout << "Policy:\n";
            // fout.precision(5);
            // fout << fixed;
            // for(int i=0; i<NUM_ACTIONS; i++){
            //     if(policy[t][i] < 0){
            //         fout << "....... ";
            //     }
            //     else{
            //         fout << policy[t][i] << ' ';
            //     }
            //     if(i % numVals == numVals-1){
            //         fout << '\n';
            //     }
            // }
            // fout << '\n';
            // fout << "Action: ";
            // for(int i=0; i<NUM_ACTIONS; i++){
            //     fout << actions[t][i] << ' ';
            //     if(i % numVals == numVals-1){
            //         fout << '\n';
            //     }
            // }
            // fout << '\n';
            // // fout << "Action: " << action << '\n';
            // fout << "Reward: " << reward[t] << "\n\n";
            // fout.close();
        }
        rollOutRewardSum += reward[t];
        if(env.endState){
            rollOutLength = t+1;
            break;
        }
    }
    double value = 0;
    for(int t=rollOutLength-1; t>=0; t--){
        value *= DISCOUNT_FACTOR;
        value += reward[t];
        double scale = value - valueMean[t];
        valueMean[t] += (value - valueMean[t]) * meanUpdate;
        for(auto a : states[t].validActions()){
            if(actions[t][a] != -1){ // OPTIMIZATION TO CHECK WHETHER ACTION WAS COUNTED.
                seq.outputs[t].gradient[a] = (policy[t][a]-actions[t][a]) * scale;
            }
        }
        // seq.outputs[t].gradient[actions[t]] = (policy[t][actions[t]] - 1) * scale;
        seq.backwardPassUnit(t);
    }
}

void PG::computeSoftmax(double* weights, double* policy, vector<int> validActions){
    for(int i=0; i<NUM_ACTIONS; i++){
        policy[i] = -1;
    }
    double maxWeight = -1e+08;
    for(auto a : validActions){
        maxWeight = max(maxWeight, weights[a]);
    }
    double sum = 0;
    for(auto a : validActions){
        assert(!isnan(weights[a]));
        policy[a] = exp(weights[a] - maxWeight);
        sum += policy[a];
    }
    for(auto a : validActions){
        policy[a] /= sum;
    }
}

void PG::computeSigmoid(double* weights, double* policy, vector<int> validActions){
    for(int i=0; i<NUM_ACTIONS; i++){
        policy[i] = -1;
    }
    for(auto a : validActions){
        policy[a] = 1/(1+exp(-weights[a]));
        assert(!isnan(weights[a]));
        assert(!isnan(policy[a]));
    }
}

void PG::train(){
    double rewardSum = 0;
    int evalPeriod = 10000;
    for(int i=0; i<TIME_HORIZON; i++){
        valueMean[i] = 0;
    }
    ofstream fout(fileOut);
    fout << "Beginning training\n";
    fout << "Learn rate: " << learnRate << '\n';
    fout.close();
    for(int iter=0; iter<300000; iter++){
        rollOut();
        rewardSum += rollOutRewardSum;
        if(iter % batchSize == 0){
            seq.paramStore.updateParams(learnRate / batchSize, momentum, regRate);
        }
        if(iter % evalPeriod == 0 && iter > 0){
            ofstream fout(fileOut, ios::app);
            fout<<"Iter: "<<iter<< ". Average Reward: "<<(rewardSum / evalPeriod)<<'\n';
            fout.close();
            // if(rewardSum / evalPeriod > 4){
            //     setLearnRate(0.005);
            // }
            rewardSum = 0;
        }
    }
    finalReward = rewardSum / evalPeriod;
}

void PG::multRollOut(int numRolls){
    double sum = 0;
    for(int iter=0; iter<numRolls; iter++){
        rollOut(false);
        sum += rollOutRewardSum;
    }
    rollOutRewardSum = sum;
}

void PG::trainParallel(int evalPeriod, int numIter){

    assert(batchSize % numSubThreads == 0);

    // allocate subtrainers

    PG subTrainers[numSubThreads];
    thread* subThreads[numSubThreads];

    double rewardSum = 0;
    for(int i=0; i<TIME_HORIZON; i++){
        valueMean[i] = 0;
    }
    ofstream fout(fileOut);
    fout << "Beginning training\n";
    fout << "Learn rate: " << learnRate << '\n';
    fout.close();
    // ofstream errOut("err.out");
    // errOut.close();

    ofstream resultOut("results.out");

    unsigned long start_time = time(0);

    for(int iter=0; iter<numIter; iter++){

        // run rollouts on each thread

        for(int i=0; i<numSubThreads; i++){
            subTrainers[i].seq.paramStore.copyParams(&seq.paramStore);
            subTrainers[i].seq.paramStore.resetGradient();
            for(int j=0; j<TIME_HORIZON; j++){
                subTrainers[i].valueMean[j] = valueMean[j];
            }
            subThreads[i] = new thread(&PG::multRollOut, &subTrainers[i], batchSize/numSubThreads);
        }

        for(int j=0; j<TIME_HORIZON; j++){
            valueMean[j] = 0;
        }
        for(int i=0; i<numSubThreads; i++){
            subThreads[i]->join();
            rewardSum += subTrainers[i].rollOutRewardSum / batchSize;
            seq.paramStore.accumulateGradient(&subTrainers[i].seq.paramStore);
            for(int j=0; j<TIME_HORIZON; j++){
                valueMean[j] += subTrainers[i].valueMean[j] / batchSize;
            }
        }
        seq.paramStore.updateParams(learnRate / batchSize, momentum, regRate);
        if(iter % evalPeriod == 0 && iter > 0){
            ofstream fout(fileOut, ios::app);
            fout<<"Iter: "<<iter<< ". Average Reward: "<<(rewardSum / evalPeriod) << " Time stamp: " << (time(0) - start_time) <<'\n';
            fout.close();
            if(iter != evalPeriod){
                resultOut << ',';
            }
            resultOut << (rewardSum / evalPeriod);
            // if(rewardSum / evalPeriod > 5){
            //     setLearnRate(0.002);
            // }
            rewardSum = 0;
        }
    }
    resultOut << '\n';
    finalReward = rewardSum / evalPeriod;
}

void PG::setLearnRate(double lr){
    if(learnRate > lr){
        learnRate = lr;
        ofstream fout(fileOut, ios::app);
        fout<<"Learn Rate set to "<<lr<<'\n';
        fout.close();
    }
}