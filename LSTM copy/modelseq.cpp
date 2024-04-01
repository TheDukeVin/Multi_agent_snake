
#include "lstm.h"

ModelSeq::ModelSeq(Model structure, int T_, double initParam){
    T = T_;
    paramStore = Model(structure, NULL, new Data(structure.inputSize), new Data(structure.outputSize));
    paramStore.randomize(initParam);
    for(int i=0; i<T; i++){
        inputs.push_back(Data(paramStore.inputSize));
        outputs.push_back(Data(paramStore.outputSize));
        // expectedOutputs.push_back(new double[paramStore.outputSize]);
        // validOutput.push_back(new bool[paramStore.outputSize]);
    }

    for(int i=0; i<T; i++){
        Model* prevUnit;
        if(i > 0) prevUnit = &seq[i-1];
        else prevUnit = NULL;

        seq.push_back(Model(structure, prevUnit, &inputs[i], &outputs[i]));
    }
}

void ModelSeq::forwardPassUnit(int index){
    seq[index].copyParams(&paramStore);
    seq[index].forwardPass();
}

void ModelSeq::forwardPass(){
    for(int i=0; i<T; i++){
        forwardPassUnit(i);
    }
}

void ModelSeq::backwardPassUnit(int index){
    seq[index].backwardPass();
    paramStore.accumulateGradient(&seq[index]);
}

void ModelSeq::backwardPass(){
    for(int i=T-1; i>=0; i--){
        // for(int j=0; j<paramStore.outputSize; j++){
        //     if(validOutput[i][j]){
        //         outputs[i].gradient[j] = 2 * (outputs[i].data[j] - expectedOutputs[i][j]);
        //     }
        // }
        seq[i].backwardPass();
        paramStore.accumulateGradient(&seq[i]);
    }
}

double ModelSeq::getLoss(){
    double sum = 0;
    // for(int i=0; i<T; i++){
    //     for(int j=0; j<paramStore.outputSize; j++){
    //         if(validOutput[i][j]){
    //             sum += pow(outputs[i].data[j] - expectedOutputs[i][j], 2);
    //         }
    //     }
    // }
    assert(false);
    return sum;
}