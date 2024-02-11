
#include "lstm.h"

using namespace LSTM;

ConvLayer::ConvLayer(Data* input_, Data* output_, Shape inputShape_, Shape outputShape_, int convH_, int convW_){
    input = input_;
    output = output_;
    inputShape = inputShape_;
    outputShape = outputShape_;
    inputSize = inputShape.getSize();
    outputSize = outputShape.getSize();
    convH = convH_;
    convW = convW_;

    int weightSize = inputShape.depth * outputShape.depth * convH * convW;
    int biasSize = outputShape.depth;

    params = new Params(weightSize + biasSize);
    Data* weights = new Data(weightSize, params->params, params->gradient);
    Data* bias = new Data(biasSize, params->params + weightSize, params->gradient + weightSize);
    allNodes.push_back(new ConvNode(input, weights, bias, output, inputShape, outputShape, convH, convW));
}