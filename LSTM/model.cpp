
#include "lstm.h"

using namespace LSTM;

// Model::Model(int inputSize_){
//     inputSize = inputSize_;
//     lastSize = inputSize_;
//     lastAct = new Data(inputSize);
//     activations.push_back(lastAct);
// }

Model::Model(Shape inputShape_){
    inputShape = inputShape_;
    if(inputShape.type == 0){
        inputSize = inputShape.size;
    }
    else{
        inputSize = inputShape.depth * inputShape.height * inputShape.width;
    }
    lastAct = new Data(inputSize);
    lastShape = inputShape;
    activations.push_back(lastAct);
}

void Model::addConv(Shape shape, int convHeight, int convWidth){
    outputSize = shape.depth * shape.height * shape.width;
    Data* newAct = new Data(outputSize);
    layers.push_back(new ConvLayer(lastAct, newAct, lastShape, shape, convHeight, convWidth));
    lastAct = newAct;
    lastShape = shape;
}

void Model::addPool(Shape shape){
    outputSize = shape.depth * shape.height * shape.width;
    Data* newAct = new Data(outputSize);
    layers.push_back(new PoolLayer(lastAct, newAct, lastShape, shape));
    lastAct = newAct;
    lastShape = shape;
}

void Model::addLSTM(int outputSize_){
    outputSize = outputSize_;
    Data* newAct = new Data(outputSize_);
    layers.push_back(new LSTMLayer(lastAct, newAct, NULL));
    activations.push_back(newAct);
    lastAct = newAct;
    lastShape = Shape(outputSize_);
}

void Model::addDense(int outputSize_){
    outputSize = outputSize_;
    Data* newAct = new Data(outputSize_);
    layers.push_back(new Dense(lastAct, newAct));
    activations.push_back(newAct);
    lastAct = newAct;
    lastShape = Shape(outputSize_);
}

void Model::addOutput(int outputSize_){
    outputSize = outputSize_;
    Data* newAct = new Data(outputSize_);
    layers.push_back(new PolicyOutput(lastAct, newAct));
    activations.push_back(newAct);
    lastAct = newAct;
}

Model::Model(Model* structure, Model* prevModel, Data* input, Data* output){
    inputSize = structure->inputSize;
    inputShape = structure->inputShape;
    outputSize = structure->outputSize;
    inputSize = structure->inputSize;
    
    activations.push_back(input);
    lastAct = input;
    for(int i=0; i<structure->layers.size(); i++){
        Data* newAct;
        if(i == structure->layers.size()-1){
            newAct = output;
        }
        else{
            newAct = new Data(structure->layers[i]->outputSize);
        }
        activations.push_back(newAct);
        // Cast derived classes before base classes
        if(dynamic_cast<LSTMLayer*>(structure->layers[i]) != NULL){
            LSTMLayer* prevLSTM;
            if(prevModel == NULL) prevLSTM = NULL;
            else prevLSTM = dynamic_cast<LSTMLayer*>(prevModel->layers[i]);
            layers.push_back(new LSTMLayer(lastAct, newAct, prevLSTM));
        }
        else if(dynamic_cast<PolicyOutput*>(structure->layers[i]) != NULL){
            layers.push_back(new PolicyOutput(lastAct, newAct));
        }
        else if(dynamic_cast<Dense*>(structure->layers[i]) != NULL){
            layers.push_back(new Dense(lastAct, newAct));
        }
        else if(dynamic_cast<ConvLayer*>(structure->layers[i]) != NULL){
            ConvLayer* conv = dynamic_cast<ConvLayer*>(structure->layers[i]);
            layers.push_back(new ConvLayer(lastAct, newAct, conv->inputShape,
                conv->outputShape, conv->convH, conv->convW));
        }
        else if(dynamic_cast<PoolLayer*>(structure->layers[i]) != NULL){
            PoolLayer* pool = dynamic_cast<PoolLayer*>(structure->layers[i]);
            layers.push_back(new PoolLayer(lastAct, newAct, pool->inputShape, pool->outputShape));
        }
        else{
            assert(false);
        }
        // cout<<"Layer " << i << '\n';
        // cout << layers[i]->inputSize << '\n';
        lastAct = newAct;
    }
    // cout<<"Input: " << inputSize << '\n';
    assert(layers.size() == structure->layers.size());
}

void Model::copyParams(Model* m){
    // cout<<"HELLO\n";
    for(int i=0; i<layers.size(); i++){
        // cout<<"Layer " << i << '\n';
        // cout << "Layer size: " << layers[i]->inputSize << '\n';
        // cout << "Struct layer size: " << m->layers[i]->inputSize << '\n';
        layers[i]->params->copy(m->layers[i]->params);
    }
}

void Model::copyAct(Model* m){
    for(int i=0; i<layers.size(); i++){
        layers[i]->copyAct(m->layers[i]);
    }
}

void Model::randomize(double scale){
    for(int i=0; i<layers.size(); i++){
        layers[i]->params->randomize(scale);
    }
}

void Model::forwardPass(){
    for(int i=0; i<layers.size(); i++){
        layers[i]->forwardPass();
    }
}

void Model::backwardPass(){
    for(int i=layers.size()-1; i>=0; i--){
        // cout << "Back Layer " << i << '\n';
        layers[i]->backwardPass();
    }
}

void Model::resetGradient(){
    for(int i=0; i<layers.size(); i++){
        layers[i]->params->resetGradient();
    }
}

void Model::accumulateGradient(Model* m){
    for(int i=0; i<layers.size(); i++){
        layers[i]->params->accumulateGradient(m->layers[i]->params);
    }
}

void Model::updateParams(double scale, double momentum, double regRate){
    for(int i=0; i<layers.size(); i++){
        layers[i]->params->update(scale, momentum, regRate);
    }
}

void Model::save(string fileOut){
    ofstream fout(fileOut);
    for(int i=0; i<layers.size(); i++){
        for(int j=0; j<layers[i]->params->size; j++){
            fout << layers[i]->params->params[j] << ' ';
        }
    }
}

void Model::load(string fileIn){
    ifstream fin(fileIn);
    for(int i=0; i<layers.size(); i++){
        for(int j=0; j<layers[i]->params->size; j++){
            fin >> layers[i]->params->params[j];
        }
    }
}