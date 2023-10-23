
/*
g++ -O2 -std=c++11 -pthread main.cpp modelseq.cpp model.cpp layer.cpp lstm.cpp policy.cpp params.cpp node.cpp

-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment

rsync -r LSTM kevindu@login.rc.fas.harvard.edu:./MultiagentSnake
rsync -r kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/LSTM/net.out LSTM

*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <cassert>

#ifndef lstm_h
#define lstm_h
using namespace std;

int sampleDist(double* dist, int N);

class Data{
public:
    int size;
    double* data;
    double* gradient;

    Data(){}
    Data(int size_);
    Data(int size_, double* data_, double* gradient_);

    void resetGradient();
};

class Node{
public:
    Data* i1;
    Data* i2;
    Data* o;

    Node(){}
    Node(Data* i1_, Data* i2_, Data* o_);

    virtual void forwardPass() = 0;
    virtual void backwardPass() = 0;
};

class ConcatNode : public Node{
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class AdditionNode : public Node{
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class MultiplicationNode : public Node{
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class MatMulNode : public Node{ // (m x n) matrix times (n x 1) vector -> (m x 1) vector
public:
    using Node::Node;
    void forwardPass();
    void backwardPass();
};

class UnitaryNode : public Node{
public:
    string operation;

    UnitaryNode(Data* i1_, Data* o_, string op);
    
    void forwardPass();
    void backwardPass();

    double nonlinear(double x);
    double dnonlinear(double x);
};

class Shape{
public:
    int type; // 0 = linear data. 1 = grid data
    int size;
    int height, width, depth;
    Shape(){}
    Shape(int size_){
        type = 0; size = size_;
    }
    Shape(int h, int w, int d){
        type = 1; height = h; width = w; depth = d;
    }
    int getSize(){
        if(type == 0) return size;
        else return height * width * depth;
    }
};

class ConvNode : public Node{
public:
    /*
    i1 = Input tensor
    i2 = Convolutional filter
    bias
    o = Output
    */
    Data* bias;
    Shape input;
    Shape output;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int w1, w2, w3;
    
    ConvNode(Data* i1_, Data* i2_, Data* bias_, Data* o_, Shape input_, Shape output_, int convH, int convW);
    void forwardPass();
    void backwardPass();

    double nonlinear(double x); // f
    double dinvnonlinear(double x); // f'(f^-1)
};

class PoolNode : public Node{
public:
    Shape input;
    Shape output;
    int* maxIndices;

    PoolNode(Data* i1_, Data* o_, Shape input_, Shape output_);
    void forwardPass();
    void backwardPass();
};

class Params{
public:
    int size;
    double* params;
    double* gradient;

    Params(){}
    Params(int size_);
    void randomize(double scale);
    void copy(Params params_);
    void accumulateGradient(Params params_);
    void update(double scale, double momentum, double regRate);
    void resetGradient();
};

class Layer{
protected:
    vector<Data*> allHiddenData;
    vector<Node*> allNodes;

    Data* addData(int size);
    void resetGradient();
    
public:
    Params params;

    int inputSize;
    int outputSize;

    Data* input;
    Data* output;

    Layer(){}

    void forwardPass(); // resets gradient of all data
    void backwardPass();

    virtual void vf(){};
};

class LSTM : public Layer{
public:
    Data* cell;

    // Looks at previous unit's output and cell.
    LSTM(int size); // empty LSTM to start the chain
    LSTM(Data* input_, Data* output_, LSTM* prevUnit);

    double nonlinear(double x);
    double dinvnonlinear(double x);
};

class Dense : public Layer{
protected:
    void setupLayer(Data* input_, Data* output_, string operation);

public:
    Dense(){}
    Dense(Data* input_, Data* output_);
};

class PolicyOutput : public Dense{
public:
    PolicyOutput(Data* input_, Data* output_);
};

class ConvLayer : public Layer{
public:
    Shape inputShape; // Store data shapes as not readable from data entries
    Shape outputShape;
    int convH, convW;

    ConvLayer(Data* input_, Data* output_, Shape inputShape_, Shape outputShape_, int convH_, int convW_);
};

class PoolLayer : public Layer{
public:
    Shape inputShape;
    Shape outputShape;

    PoolLayer(Data* input_, Data* output_, Shape inputShape_, Shape outputShape_);
};

class Model{
private:
    Shape lastShape; // used to initialize model
    Data* lastAct;

public:
    vector<Layer*> layers;
    vector<Data*> activations;

    Shape inputShape;
    int inputSize;
    int outputSize;

    Model(){}

    // Construct a model structure
    Model(Shape inputShape);
    void addConv(Shape shape, int convHeight, int convWidth);
    void addPool(Shape shape);
    void addLSTM(int outputSize_);
    void addDense(int outputSize_);
    void addOutput(int outputSize_);

    // Define an active Model unit from given structure
    Model(Model structure, Model* prevModel, Data* input, Data* output);

    void copyParams(Model* m);
    void randomize(double scale);

    void forwardPass();
    void backwardPass();

    void resetGradient();
    void accumulateGradient(Model* m);
    void updateParams(double scale, double momentum, double regRate);

    void save(string fileOut);
    void load(string fileIn);
};

class ModelSeq{
public:
    int T;
    vector<Model> seq;
    vector<Data> inputs;
    vector<Data> outputs;
    // vector<double*> expectedOutputs;
    // vector<bool*> validOutput;
    Model paramStore;

    ModelSeq(){}
    ModelSeq(Model structure, int T_, double initParam);
    void forwardPassUnit(int index);
    void forwardPass();
    void backwardPassUnit(int index);
    void backwardPass();

    double getLoss();
};

#endif