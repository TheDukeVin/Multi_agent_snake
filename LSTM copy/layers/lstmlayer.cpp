
#include "lstm.h"

using namespace LSTM;

LSTMLayer::LSTMLayer(int size){
    output = new Data(size);
    cell = new Data(size);
    for(int i=0; i<size; i++){
        output->data[i] = output->gradient[i] = 0;
        cell->data[i] = cell->gradient[i] = 0;
    }
}

LSTMLayer::LSTMLayer(Data* input_, Data* output_, LSTMLayer* prevUnit){
    input = input_;
    output = output_;
    inputSize = input->size;
    outputSize = output->size;
    params = new Params(4 * ((inputSize + outputSize + 1) * outputSize));
    if(prevUnit == NULL){
        prevUnit = new LSTMLayer(outputSize);
    }

    // Initialize Data and Nodes

    Data* XH = addData(inputSize + outputSize);
    allNodes.push_back(new ConcatNode(input, prevUnit->output, XH));

    int weightSize = (inputSize + outputSize) * outputSize;
    int biasSize = outputSize;

    Data* weights[4];
    Data* bias[4];
    vector<Data*> linComb;
    for(int i=0; i<4; i++){
        weights[i] = new Data(weightSize, params->params + (weightSize + biasSize)*i,
                                          params->gradient + (weightSize + biasSize)*i);
        bias[i] = new Data(biasSize, params->params + (weightSize + biasSize)*i + weightSize,
                                     params->gradient + (weightSize + biasSize)*i + weightSize);
        Data* mult_result = addData(outputSize);
        allNodes.push_back(new MatMulNode(weights[i], XH, mult_result));
        Data* sum_result = addData(outputSize);
        allNodes.push_back(new AdditionNode(mult_result, bias[i], sum_result));
        linComb.push_back(sum_result);
    }
    Data* F = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[0], F, "sigmoid"));
    Data* G = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[1], G, "sigmoid"));
    Data* H = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[2], H, "sigmoid"));
    Data* C = addData(outputSize);
    allNodes.push_back(new UnitaryNode(linComb[3], C, "tanh"));

    Data* C1 = addData(outputSize);
    allNodes.push_back(new MultiplicationNode(F, prevUnit->cell, C1));
    Data* C2 = addData(outputSize);
    allNodes.push_back(new MultiplicationNode(G, C, C2));
    cell = addData(outputSize);
    allNodes.push_back(new AdditionNode(C1, C2, cell));

    Data* feedback = addData(outputSize);
    allNodes.push_back(new UnitaryNode(cell, feedback, "tanh"));

    allNodes.push_back(new MultiplicationNode(H, feedback, output));
}