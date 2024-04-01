
/*
g++ -O2 -std=c++11 -pthread main.cpp modelseq.cpp model.cpp layer.cpp lstm.cpp policy.cpp params.cpp node.cpp

g++ -O2 -std=c++11 -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment -I "/Users/kevindu/Desktop/Employment/Multiagent Snake Research/multiagent_snake/LSTM" main.cpp model.cpp PVUnit.cpp layer.cpp layers/lstmlayer.cpp layers/policy.cpp layers/conv.cpp layers/pool.cpp params.cpp node.cpp

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
#include <unordered_set>

#ifndef lstm_h
#define lstm_h
using namespace std;

namespace LSTM
{

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
    void copy(Data* d);

    ~Data(){
        // cout << "Deleting data\n";
        delete[] data;
        delete[] gradient;
    }
};

class Node{
public:
    Data* i1 = NULL;
    Data* i2 = NULL;
    Data* o = NULL;

    Node(){}
    Node(Data* i1_, Data* i2_, Data* o_);

    virtual void forwardPass() = 0;
    virtual void backwardPass() = 0;

    virtual ~Node(){}
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

    ~PoolNode(){
        delete maxIndices;
    }
};

class Params{
public:
    const static double constexpr beta_1 = 0.9;
    const static double constexpr beta_2 = 0.999;
    const static double constexpr epsilon = 1e-08;

    int size;
    double* params;
    double* gradient;

    // For Adam optimization
    double* first_moment;
    double* second_moment;
    long numUpdates;

    Params(){}
    Params(int size_);
    void randomize(double scale);
    void copy(Params* params_);
    void accumulateGradient(Params* params_);
    void update(double scale, double momentum, double regRate);
    void resetGradient();

    ~Params(){
        // cout << "Deleting params\n";
        delete[] params;
        delete[] gradient;
        delete[] first_moment;
        delete[] second_moment;
    }
};

class Layer{
protected:
    Data* addData(int size);
    void resetGradient();
    
public:
    Params* params;
    vector<Data*> allHiddenData;
    vector<Node*> allNodes;

    int inputSize;
    int outputSize;

    Data* input;
    Data* output;

    Layer(){}

    void forwardPass(); // resets gradient of all data
    void backwardPass();

    void copyAct(Layer* l);

    virtual void vf(){};

    ~Layer(){
        // cout << "Deleting layer\n";
        delete params;
    }
};

class LSTMLayer : public Layer{
public:
    Data* cell;

    // Looks at previous unit's output and cell.
    LSTMLayer(int size); // empty LSTM to start the chain
    LSTMLayer(Data* input_, Data* output_, LSTMLayer* prevUnit);

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
    Model(Model* structure, Model* prevModel, Data* input, Data* output);

    void copyParams(Model* m);
    void copyAct(Model* m);
    void randomize(double scale);

    void forwardPass();
    void backwardPass();

    void resetGradient();
    void accumulateGradient(Model* m);
    void updateParams(double scale, double momentum, double regRate);

    void save(string fileOut);
    void load(string fileIn);

    ~Model(){
        // cout << "Deleting model\n";
    }
};

class PVUnit{
public:
    Model* commonBranch = NULL;
    Model* policyBranch;
    Model* valueBranch;
    vector<Model*> allBranches;

    Data* envInput;
    Data* commonComp;
    Data* policyOutput;
    Data* valueOutput;

    // Construct a structure for commonBranch, policyBranch, and valueBranch first
    PVUnit(){}
    void initPV();
    void setupPV();

    // Then define new instances of the structure
    PVUnit(PVUnit* structure, PVUnit* prevUnit);

    void copyParams(PVUnit* unit);
    void copyAct(PVUnit* unit);
    void randomize(double scale);

    void forwardPass();
    void backwardPass();

    void resetGradient();
    void accumulateGradient(PVUnit* unit);
    void updateParams(double scale, double momentum, double regRate);

    void save(string fileOut);
    void load(string fileIn);

    ~PVUnit(){
        cout << "Deleting PVUnit\n";
        unordered_set<Data*> allData;
        for(int i=0; i<allBranches.size(); i++){
            Model* branch = allBranches[i];
            // cout << "Iterating branch " << i << '\n';
            for(int j=0; j<branch->layers.size(); j++){
                Layer* layer = branch->layers[j];
                // cout << "Iterating Layer " << j << '\n';
                for(auto node : layer->allNodes){
                    delete node;
                }
                for(auto data : layer->allHiddenData){
                    allData.insert(data);
                }
                delete layer;
            }
            delete branch;
        }
        for(auto data : allData){
            if(data != NULL){
                // cout << data << '\n';
                delete data;
            }
        }
        cout << "Successfully deleted\n";
    }
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
}
#endif