
#include "lstm.h"

#ifndef test_h
#define test_h

class ModelTest{
public:
    LSTM::Model structure;
    vector<LSTM::Model> units;
    int T;
    vector<double*> expectedOutput;
    vector<LSTM::Data*> inputs;
    vector<LSTM::Data*> outputs;

    ModelTest(){
        structure = LSTM::Model(LSTM::Shape(5, 6, 3));
        structure.addConv(LSTM::Shape(4, 6, 2), 3, 3);
        structure.addPool(LSTM::Shape(2, 3, 2));
        structure.addDense(4);
        structure.addOutput(3);
        structure.randomize(1);
        T = 3;
        for(int i=0; i<T; i++){
            LSTM::Model* prevUnit;
            // if(i > 0) prevUnit = &units[i-1];
            // else prevUnit = NULL;
            prevUnit = NULL;

            inputs.push_back(new LSTM::Data(structure.inputSize));
            outputs.push_back(new LSTM::Data(structure.outputSize));
            units.push_back(LSTM::Model(structure, prevUnit, inputs[i], outputs[i]));
        }
        forwardPass();
    }

    void generateRandom(){
        for(auto out : expectedOutput){
            delete out;
        }
        expectedOutput = vector<double*>();
        for(int i=0; i<T; i++){
            for(int j=0; j<structure.inputSize; j++){
                inputs[i]->data[j] = 2 * (double) rand() / RAND_MAX - 1;
            }
            expectedOutput.push_back(new double[structure.outputSize]);
            for(int j=0; j<structure.outputSize; j++){
                expectedOutput[i][j] = 2 * (double) rand() / RAND_MAX - 1;
            }
        }
    }

    double getLoss(){
        double sum = 0;
        for(int i=0; i<T; i++){
            for(int j=0; j<structure.outputSize; j++){
                sum += pow(outputs[i]->data[j] - expectedOutput[i][j], 2);
            }
        }
        return sum;
    }

    void forwardPass(){
        for(int i=0; i<T; i++){
            units[i].copyParams(&structure);
            units[i].forwardPass();
        }
    }

    void test(){
        generateRandom();
        structure.resetGradient();
        forwardPass();
        for(int i=0; i<T; i++){
            for(int j=0; j<structure.outputSize; j++){
                outputs[i]->gradient[j] = 2 * (outputs[i]->data[j] - expectedOutput[i][j]);
            }
        }
        for(int i=T-1; i>=0; i--){
            units[i].backwardPass();
            structure.accumulateGradient(&units[i]);
        }
        double initLoss = getLoss();
        double epsilon = 1e-07;
        double tol = 1e-04;
        for(int i=0; i<structure.layers.size(); i++){
            for(int j=0; j<structure.layers[i]->params->size; j++){
                structure.layers[i]->params->params[j] += epsilon;
                forwardPass();
                double newLoss = getLoss();
                structure.layers[i]->params->params[j] -= epsilon;
                double derivative = (newLoss - initLoss) / epsilon;
                assert(abs(derivative - structure.layers[i]->params->gradient[j]) < tol);
                cout << derivative << ' ' << structure.layers[i]->params->gradient[j] << '\n';
            }
        }
    }
};

class PVTest{
public:
    LSTM::PVUnit structure;
    vector<LSTM::PVUnit> units;
    int T;
    vector<double*> expectedPolicy;
    vector<double*> expectedValue;

    PVTest(){
        structure.commonBranch = new LSTM::Model(LSTM::Shape(5, 5, 3));
        structure.commonBranch->addConv(LSTM::Shape(4, 4, 2), 2, 2);
        structure.commonBranch->addPool(LSTM::Shape(2, 2, 2));
        structure.initPV();
        structure.policyBranch->addLSTM(4);
        structure.policyBranch->addOutput(3);
        structure.valueBranch->addLSTM(5);
        structure.valueBranch->addOutput(1);
        structure.randomize(1);
        T = 3;
        for(int i=0; i<T; i++){
            LSTM::PVUnit* prevUnit;
            if(i > 0) prevUnit = &units[i-1];
            else prevUnit = NULL;

            units.push_back(LSTM::PVUnit(structure, prevUnit));
        }
        forwardPass();
    }

    void generateRandom(){
        for(auto out : expectedPolicy){
            delete out;
        }
        for(auto out : expectedValue){
            delete out;
        }
        expectedPolicy = vector<double*>();
        expectedValue = vector<double*>();
        for(int i=0; i<T; i++){
            for(int j=0; j<structure.commonBranch->inputSize; j++){
                units[i].envInput->data[j] = 2 * (double) rand() / RAND_MAX - 1;
            }
            expectedPolicy.push_back(new double[structure.policyBranch->outputSize]);
            expectedValue.push_back(new double[structure.valueBranch->outputSize]);
            for(int j=0; j<structure.policyBranch->outputSize; j++){
                expectedPolicy[i][j] = 2 * (double) rand() / RAND_MAX - 1;
            }
            for(int j=0; j<structure.valueBranch->outputSize; j++){
                expectedValue[i][j] = 2 * (double) rand() / RAND_MAX - 1;
            }
        }
    }

    double getLoss(){
        double sum = 0;
        for(int i=0; i<T; i++){
            for(int j=0; j<structure.policyBranch->outputSize; j++){
                sum += pow(units[i].policyOutput->data[j] - expectedPolicy[i][j], 2);
            }
            for(int j=0; j<structure.valueBranch->outputSize; j++){
                sum += pow(units[i].valueOutput->data[j] - expectedValue[i][j], 2);
            }
        }
        return sum;
    }

    void forwardPass(){
        for(int i=0; i<T; i++){
            units[i].copyParams(&structure);
            units[i].forwardPass();
        }
    }

    void test(){
        generateRandom();
        structure.resetGradient();
        forwardPass();
        for(int i=0; i<T; i++){
            for(int j=0; j<structure.policyBranch->outputSize; j++){
                units[i].policyOutput->gradient[j] = 2 * (units[i].policyOutput->data[j] - expectedPolicy[i][j]);
            }
            for(int j=0; j<structure.valueBranch->outputSize; j++){
                units[i].valueOutput->gradient[j] = 2 * (units[i].valueOutput->data[j] - expectedValue[i][j]);
            }
        }
        for(int i=T-1; i>=0; i--){
            units[i].backwardPass();
            structure.accumulateGradient(&units[i]);
        }
        double initLoss = getLoss();
        double epsilon = 1e-07;
        double tol = 1e-04;
        for(int l=0; l<structure.allBranches.size(); l++){
            for(int i=0; i<structure.allBranches[l]->layers.size(); i++){
                for(int j=0; j<structure.allBranches[l]->layers[i]->params->size; j++){
                    structure.allBranches[l]->layers[i]->params->params[j] += epsilon;
                    forwardPass();
                    double newLoss = getLoss();
                    structure.allBranches[l]->layers[i]->params->params[j] -= epsilon;
                    double derivative = (newLoss - initLoss) / epsilon;
                    assert(abs(derivative - structure.allBranches[l]->layers[i]->params->gradient[j]) < tol);
                }
            }
        }
    }
};

#endif