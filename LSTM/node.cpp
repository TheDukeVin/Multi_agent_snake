
#include "lstm.h"

using namespace LSTM;

Data::Data(int size_){
    size = size_;
    data = new double[size];
    gradient = new double[size];
    for(int i=0; i<size; i++){
        data[i] = gradient[i] = 0;
    }
}

Data::Data(int size_, double* data_, double* gradient_){
    size = size_;
    data = data_;
    gradient = gradient_;
}

void Data::resetGradient(){
    for(int i=0; i<size; i++){
        gradient[i] = 0;
    }
}

void Data::copy(Data* d){
    for(int i=0; i<size; i++){
        data[i] = d->data[i];
    }
}

Node::Node(Data* i1_, Data* i2_, Data* o_){
    i1 = i1_; i2 = i2_; o = o_;
}

UnitaryNode::UnitaryNode(Data* i1_, Data* o_, string op){
    i1 = i1_; o = o_; operation = op;
}

double sigmoid(double x){
    return 1/(1 + exp(-x));
}

double tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double UnitaryNode::nonlinear(double x){
    if(operation == "sigmoid"){
        return sigmoid(x);
    }
    if(operation == "tanh"){
        return tanh(x);
    }
    if(operation == "identity"){
        return x;
    }
    if(operation == "relu"){
        if(x < 0) return 0;
        return x;
    }
    if(operation == "leakyRelu"){
        if(x < 0) return 0.1*x;
        return x;
    }
    assert(false);
}

double UnitaryNode::dnonlinear(double x){
    if(operation == "sigmoid"){
        double s = sigmoid(x);
        return s * (1 - s);
    }
    if(operation == "tanh"){
        double t = tanh(x);
        return 1 - t * t;
    }
    if(operation == "identity"){
        return 1;
    }
    if(operation == "relu"){
        if(x < 0) return 0;
        return 1;
    }
    if(operation == "leakyRelu"){
        if(x < 0) return 0.1;
        return 1;
    }
    assert(false);
}

// Define forward and backward pass operations

void ConcatNode::forwardPass(){
    for(int i=0; i<i1->size; i++){
        o->data[i] = i1->data[i];
    }
    for(int i=0; i<i2->size; i++){
        o->data[i + i1->size] = i2->data[i];
    }
}

void ConcatNode::backwardPass(){
    for(int i=0; i<i1->size; i++){
        i1->gradient[i] += o->gradient[i];
    }
    for(int i=0; i<i2->size; i++){
        i2->gradient[i] += o->gradient[i + i1->size];
    }
}

void AdditionNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        o->data[i] = i1->data[i] + i2->data[i];
    }
}

void AdditionNode::backwardPass(){
    for(int i=0; i<o->size; i++){
        i1->gradient[i] += o->gradient[i];
        i2->gradient[i] += o->gradient[i];
    }
}

void MultiplicationNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        o->data[i] = i1->data[i] * i2->data[i];
    }
}

void MultiplicationNode::backwardPass(){
    for(int i=0; i<o->size; i++){
        i1->gradient[i] += o->gradient[i] * i2->data[i];
        i2->gradient[i] += o->gradient[i] * i1->data[i];
    }
}

void MatMulNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        double sum = 0;
        for(int j=0; j<i2->size; j++){
            sum += i1->data[i*i2->size + j] * i2->data[j];
        }
        o->data[i] = sum;
    }
}

void MatMulNode::backwardPass(){
    for(int j=0; j<i2->size; j++){
        double sum = 0;
        for(int i=0; i<o->size; i++){
            i1->gradient[i*i2->size + j] += i2->data[j] * o->gradient[i];
            sum += i1->data[i*i2->size + j] * o->gradient[i];
        }
        i2->gradient[j] += sum;
    }
}

void UnitaryNode::forwardPass(){
    for(int i=0; i<o->size; i++){
        o->data[i] = nonlinear(i1->data[i]);
    }
}

void UnitaryNode::backwardPass(){
    for(int i=0; i<o->size; i++){
        i1->gradient[i] += dnonlinear(i1->data[i]) * o->gradient[i];
    }
}

ConvNode::ConvNode(Data* i1_, Data* i2_, Data* bias_, Data* o_, Shape input_, Shape output_, int convH, int convW){
    i1 = i1_; i2 = i2_; bias = bias_; o = o_;

    input = input_;
    output = output_;
    convHeight = convH;
    convWidth = convW;
    
    shiftr = (input.height - output.height - convHeight + 1) / 2;
    shiftc = (input.width - output.width - convWidth + 1) / 2;
    w1 = output.depth * convHeight * convWidth;
    w2 = convHeight * convWidth;
    w3 = convWidth;
}

void ConvNode::forwardPass(){
    double sum;
    for(int j=0; j<output.depth; j++){
        for(int x=0; x<output.height; x++){
            for(int y=0; y<output.width; y++){
                // cout<<"ORBIT " << j<<' '<<x<<' '<<y<<'\n';
                sum = bias->data[j];
                // cout<<sum<<'\n';
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < input.height && inputc >= 0 && inputc < input.width){
                            for(int i=0; i<input.depth; i++){
                                // cout << r << ' ' << c << ' ' << i << '\n';
                                // cout<<"Index: " << i*input.height*input.width + inputr*input.width + inputc << ' ' << i*w1 + j*w2 + r*w3 + c<<'\n';
                                // cout<<"Input value: " << i1->data[i*input.height*input.width + inputr*input.width + inputc] <<'\n';
                                // cout<<"Weight value: " << i2->data[i*w1 + j*w2 + r*w3 + c]<<'\n';
                                sum += i1->data[i*input.height*input.width + inputr*input.width + inputc] * i2->data[i*w1 + j*w2 + r*w3 + c];
                            }
                        }
                    }
                }
                // cout<<"Output Index: " << j*output.height*output.width + x*output.width + y << '\n'; 
                o->data[j*output.height*output.width + x*output.width + y] = nonlinear(sum);
                // cout<<"Succ\n";
            }
        }
    }
}

void ConvNode::backwardPass(){
    for(int j=0; j<output.depth; j++){
        for(int x=0; x<output.height; x++){
            for(int y=0; y<output.width; y++){
                int outIndex = j*output.height*output.width + x*output.width + y;
                double grad = dinvnonlinear(o->data[outIndex]) * o->gradient[outIndex];
                bias->gradient[j] += grad;
                for(int r=0; r<convHeight; r++){
                    for(int c=0; c<convWidth; c++){
                        int inputr = x + r + shiftr;
                        int inputc = y + c + shiftc;
                        if(inputr >= 0 && inputr < input.height && inputc >= 0 && inputc < input.width){
                            for(int i=0; i<input.depth; i++){
                                int inpIndex = i*input.height*input.width + inputr*input.width + inputc;
                                int filtIndex = i*w1 + j*w2 + r*w3 + c;
                                i1->gradient[inpIndex] += grad * i2->data[filtIndex];
                                i2->gradient[filtIndex] += grad * i1->data[inpIndex];
                                // sum += i1->data[i*input.height*input.width + inputr*input.width + inputc] * i2->data[i*w1 + j*w2 + r*w3 + c];
                            }
                        }
                    }
                }
            }
        }
    }
}

double ConvNode::nonlinear(double x){
    if(x < 0) return 0;
    return x;
}

double ConvNode::dinvnonlinear(double x){
    if(x < 1e-15) return 0;
    return 1;
}


PoolNode::PoolNode(Data* i1_, Data* o_, Shape input_, Shape output_){
    i1 = i1_; o = o_;

    input = input_;
    output = output_;

    maxIndices = new int[output.getSize()];
}

void PoolNode::forwardPass(){
    double maxVal,candVal;
    int maxIndex;
    int index;
    for(int j=0; j<output.depth; j++){
        for(int x=0; x<output.height; x++){
            for(int y=0; y<output.width; y++){
                maxVal = -1e+10;
                maxIndex = -1;
                for(int r=0; r<2; r++){
                    for(int c=0; c<2; c++){
                        index = j*input.height*input.width + (2*x+r)*input.width + (2*y+c);
                        candVal = i1->data[index];
                        if(maxVal < candVal){
                            maxVal = candVal;
                            maxIndex = index;
                        }
                    }
                }
                // assert(maxIndex >= 0);
                index = j*output.height*output.width + x*output.width + y;
                o->data[index] = maxVal;
                maxIndices[index] = maxIndex;
            }
        }
    }
}

void PoolNode::backwardPass(){
    // cout << input.depth << ' ' << input.height << ' ' << input.width << '\n';
    // cout << output.depth << ' ' << output.height << ' ' << output.width << '\n';
    // cout << this << '\n';
    // cout << maxIndices << '\n';
    for(int i=0; i<output.getSize(); i++){
        // cout << maxIndices[i] << '\n';
        // assert(0 <= maxIndices[i] && maxIndices[i] < input.getSize());
        i1->gradient[maxIndices[i]] += o->gradient[i];
    }
}