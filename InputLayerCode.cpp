//
//  InputLayerCode.cpp
//  InputLayerCode
//
//  Created by Kevin Du on 1/22/22.
//

#include "snake.h"

InputLayer::InputLayer(int outD, int outH, int outW, int convH, int convW, networkInput* input){
    outputDepth = outD;
    outputHeight = outH;
    outputWidth = outW;
    convHeight = convH;
    convWidth = convW;
    env = input;
    shiftr = (boardx - outputHeight - convHeight + 1) / 2;
    shiftc = (boardy - outputWidth - convWidth + 1) / 2;
    posShiftr = (1 - convHeight) / 2;
    posShiftc = (1 - convWidth) / 2;
    w1 = outputDepth * convHeight * convWidth;
    w2 = convHeight * convWidth;
    w3 = convWidth;
    
    numWeights = (6 * numAgents + 1) * outputDepth * convHeight * convWidth;
    numBias = outputDepth;
    this->setupParams();
}

void InputLayer::pass(){
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                double output = bias[j];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < boardx && inputc >= 0 && inputc < boardy){
                            int input = env->snake[inputr][inputc];
                            if(input != -1){
                                output += weights[input*w1 + j*w2 + r*w3 + c];
                            }
                        }
                    }
                }
                outputs[j*outputHeight*outputWidth + x*outputWidth + y] = nonlinear(output);
            }
        }
    }
}

void InputLayer::accumulateGradient(){
    double sum;
    for(int j=0; j<outputDepth; j++){
        sum = 0;
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                sum += Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
            }
        }
        Dbias[j] += sum;
    }
    for(int j=0; j<outputDepth; j++){
        for(int x=0; x<outputHeight; x++){
            for(int y=0; y<outputWidth; y++){
                double Doutput = Doutputs[j*outputHeight*outputWidth + x*outputWidth + y];
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < boardx && inputc >= 0 && inputc < boardy){
                            int input = env->snake[inputr][inputc];
                            if(input != -1){
                                Dweights[input*w1 + j*w2 + r*w3 + c] += Doutput;
                            }
                        }
                    }
                }
            }
        }
    }
}
