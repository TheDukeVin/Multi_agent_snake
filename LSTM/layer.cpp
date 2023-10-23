
#include "lstm.h"

Data* Layer::addData(int size){
    Data* data = new Data(size);
    allHiddenData.push_back(data);
    return data;
}

void Layer::forwardPass(){
    resetGradient();
    for(int i=0; i<allNodes.size(); i++){
        allNodes[i]->forwardPass();
    }
}

void Layer::backwardPass(){
    for(int i=allNodes.size()-1; i>=0; i--){
        allNodes[i]->backwardPass();
    }
}

void Layer::resetGradient(){
    input->resetGradient();
    output->resetGradient();
    for(int i=0; i<allHiddenData.size(); i++){
        allHiddenData[i]->resetGradient();
    }
    params.resetGradient();
}