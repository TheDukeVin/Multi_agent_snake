
#include "lstm.h"

using namespace LSTM;

PoolLayer::PoolLayer(Data* input_, Data* output_, Shape inputShape_, Shape outputShape_){
    input = input_;
    output = output_;
    inputShape = inputShape_;
    outputShape = outputShape_;
    inputSize = inputShape.getSize();
    outputSize = outputShape.getSize();

    params = new Params(0);
    allNodes.push_back(new PoolNode(input, output, inputShape, outputShape));
}