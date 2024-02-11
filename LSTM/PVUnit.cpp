
#include "lstm.h"

using namespace LSTM;

void PVUnit::initPV(){
    policyBranch = new Model(Shape(commonBranch->outputSize));
    valueBranch = new Model(Shape(commonBranch->outputSize));
    setupPV();
}

void PVUnit::setupPV(){
    allBranches.push_back(commonBranch);
    allBranches.push_back(policyBranch);
    allBranches.push_back(valueBranch);
}

PVUnit::PVUnit(PVUnit* structure, PVUnit* prevUnit){
    // cout << "creating data\n";
    envInput = new Data(structure->commonBranch->inputSize);
    commonComp = new Data(structure->commonBranch->outputSize);
    policyOutput = new Data(structure->policyBranch->outputSize);
    valueOutput = new Data(structure->valueBranch->outputSize);
    // cout << "creating models\n";
    if(prevUnit == NULL){
        commonBranch = new Model(structure->commonBranch, NULL, envInput, commonComp);
        policyBranch = new Model(structure->policyBranch, NULL, commonComp, policyOutput);
        valueBranch = new Model(structure->valueBranch, NULL, commonComp, valueOutput);
    }
    else{
        commonBranch = new Model(structure->commonBranch, prevUnit->commonBranch, envInput, commonComp);
        policyBranch = new Model(structure->policyBranch, prevUnit->policyBranch, commonComp, policyOutput);
        valueBranch = new Model(structure->valueBranch, prevUnit->valueBranch, commonComp, valueOutput);
    }
    // cout << commonBranch->inputSize << ' ' << policyBranch->inputSize << ' ' << valueBranch->inputSize << '\n';
    setupPV();
    // for(int i=0; i<allBranches.size(); i++){
    //     cout<<"Branch " << i<<'\n';
    //     cout << allBranches[i]->inputSize << '\n';
    // }
    // cout << "Completed constructor\n";
}

void PVUnit::copyParams(PVUnit* unit){
    for(int i=0; i<allBranches.size(); i++){
        // cout<<"Branch " << i<<'\n';
        // cout << "Unit input size: " << allBranches[i]->inputSize << '\n';
        // cout << "Struct input size: " << unit->allBranches[i]->inputSize << '\n';
        allBranches[i]->copyParams(unit->allBranches[i]);
    }
}

void PVUnit::copyAct(PVUnit* unit){
    for(int i=0; i<allBranches.size(); i++){
        allBranches[i]->copyAct(unit->allBranches[i]);
    }
}

void PVUnit::randomize(double scale){
    for(int i=0; i<allBranches.size(); i++){
        allBranches[i]->randomize(scale);
    }
}

void PVUnit::forwardPass(){
    for(int i=0; i<allBranches.size(); i++){
        allBranches[i]->forwardPass();
    }
}

void PVUnit::backwardPass(){
    for(int i=allBranches.size()-1; i>=0; i--){
        // cout << "Back Branch " << i << '\n';
        allBranches[i]->backwardPass();
    }
}

void PVUnit::resetGradient(){
    for(int i=allBranches.size()-1; i>=0; i--){
        allBranches[i]->resetGradient();
    }
}

void PVUnit::accumulateGradient(PVUnit* unit){
    for(int i=allBranches.size()-1; i>=0; i--){
        allBranches[i]->accumulateGradient(unit->allBranches[i]);
    }
}

void PVUnit::updateParams(double scale, double momentum, double regRate){
    for(int i=allBranches.size()-1; i>=0; i--){
        allBranches[i]->updateParams(scale, momentum, regRate);
    }
}

void PVUnit::save(string fileOut){
    for(int i=allBranches.size()-1; i>=0; i--){
        allBranches[i]->save(fileOut);
    }
}

void PVUnit::load(string fileIn){
    for(int i=allBranches.size()-1; i>=0; i--){
        allBranches[i]->load(fileIn);
    }
}