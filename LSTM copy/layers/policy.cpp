
#include "lstm.h"

using namespace LSTM;

void Dense::setupLayer(Data* input_, Data* output_, string operation){
    input = input_;
    output = output_;
    inputSize = input->size;
    outputSize = output->size;
    params = new Params(inputSize * outputSize + outputSize);

    int weightSize = inputSize * outputSize;
    int biasSize = outputSize;
    Data* weights = new Data(weightSize, params->params, params->gradient);
    Data* bias = new Data(biasSize, params->params + weightSize, params->gradient + weightSize);
    Data* mult_result = addData(outputSize);
    allNodes.push_back(new MatMulNode(weights, input, mult_result));
    Data* add_result = addData(outputSize);
    allNodes.push_back(new AdditionNode(mult_result, bias, add_result));
    allNodes.push_back(new UnitaryNode(add_result, output, operation));
}

Dense::Dense(Data* input_, Data* output_){
    setupLayer(input_, output_, "leakyRelu");
}

PolicyOutput::PolicyOutput(Data* input_, Data* output_){
    setupLayer(input_, output_, "identity");
}