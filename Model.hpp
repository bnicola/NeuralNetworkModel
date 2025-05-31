#ifndef __MODEL_HPP__
#define __MODEL_HPP__
#include <stdint.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "Layer.hpp"
#include <random>

/*
   Code Written by : Baher Nicola
   Date            : 12/08/2023
*/

/*
    This is the Neural network core model
    This is used to create any number of layers
    to construct a neural network.

    This can be used for image classification.

    The fully connected (FC) layers are tested and functioning properly.
    The convolutional neural network has been tested along with maxpool
    layers and has proven to be working, but it is a bit slow.

    So caution should be taken when using this library's convolutional layers
*/

class Model
{
public:

  // This is a generic class struct used to generate a 
  // string label from a classifier index.
  // It is better to have an output with a string than
  // just an index of the classified object.
  struct Classify
  {
    uint32_t index;
    std::string  label;
  };
  // Our list of classes.
  std::vector<Classify> classes_;

public:
  // Constructor/Destructor
  Model(double learning_rate = 0.1);
  ~Model();

  // Creating an input layer, this is always will be the very first call 
  // in the process of creating a neural network.
  Layer* CreateInputLayer(int curr_height, int curr_width, int current_depth);

  // Creating a fully connected layer, it takes the number of nodes for that layer
  // and the activation function that will be used.
  Layer* CreateFullLayer(uint32_t numNodes, Layer::ActivationFunc func);

  // Creating a fully connected layer with a residual connection to a previous layers, it takes the number of nodes for that layer
  // and the activation function that will be used and also the layer that is going to be added.
  Layer* CreateResidualFullLayer(uint32_t numNodes, Layer::ActivationFunc func, int residual_source_layer = -1);

  // Creating a convolutional layer, it takes the depth of the layer(the number of kernels needed)
  // and the kernel size(3, 5, 7...etc), the stride used and padding. Also specify the activation 
  // function that will be used.
  Layer* CreateConvLayer(int current_depth, int kernel_size, int stride, int padding, Layer::ActivationFunc funct);

  // Creating a Max pool layer, it takes the size of the kernel (2, 3...etc.) and the stride
  // used, as there is no activation for this layer, no activation function will be used.
  Layer* CreateMaxpoolLayer(uint32_t size, uint32_t stride);

  Layer* CreateNormalizationLayer(/*uint32_t numOutputs*/);

  // Adding a class index with a string label so when calling the classify function
  // It will return a string for what it saw in the image.
  void   AddClass(uint32_t index, std::string label);

  // This is the function that giving a serialised image will return the 
  // name of what it saw.
  std::string ClassifyInput(std::vector<double> inputs);

  // Setting our learning rate for the learning process.
  void   SetLearningRate(double rate);
  // Getting our learning rate.
  double GetLearningRate();

  // Setting our drop rate for the learning process.
  void SetDropRate(double droprate);

  // Setting pruning to true so that to decrease our model parameters.
  // Also setting our minimum weight threshold for pruning.
  void SetPruning(bool pruning, double threshold);

  // This model class reads the training data structured as a list of directories
  // named after each class (directory called cat, another called dog...etc.) and each 
  // directory will contain the training images for each class. The train function will
  // read these directories and will train the model accordingly.
  uint32_t FindClassFromDir(std::string dir);

  // This function returns the index of the maximum probability of the classifier.
  // It is used at the very end layer after a feed forward operation.
  uint32_t GetMax(std::vector<double> values);

  // Reads the predictions from the last layer after a forward path
  std::vector<double> GetOutput();

  // Returns how confident our classification was.
  float GetCurrentConfidence();

  // Inject our image/data to the network
  void   InputData(std::vector<double> values);

  // Learn this data giving this inputs and our corresponding output.
  void   LearnData(std::vector<double> inputs, std::vector<double> outputs);

  // Returns our total errors for a given layer.
  // usually used for the last layer to get the total loss to see it progressing.
  double GetTotalError();

  double GetAverageError(uint32_t epochs);

  // This is used after creating a model to save its configuration.
  // This is used so as not to create a model every time we open the application
  // Just will use loadModel to see if we are previously created this model and whether 
  // it is trained or not.
  bool   SaveModel(std::string Model = "");

  // This is used when we start our application to check if we already have a model with
  // the name specified so as to load the weights and do not need to train again.
  bool   LoadModel(std::string Model = "Default");

  // This function could be used to provide the model with a csv file to train the
  // model with.
  void   TrainDataSet(std::string fileName, uint32_t epochs, uint32_t batchSize = 128, double minLearningRate = 0.01, bool scaled = true);

  // Saving the weights of the current model.
  void   SaveWeights(std::string ModelModel = "Default");

  // Loading up the weights of the current model.
  bool   LoadWeights(std::string Model = "");

  // This is for convenience only to read a csv file and the data in the file is organised as 
  // columns of input and the last column is the index of the classification.
  void   ReadCsvFile(std::string fileName, std::vector<std::vector<double>>& inputs, std::vector<uint32_t>& output_index, bool scaled = true);

  // This is for convenience only to read a csv file and test the data in the file organised as 
  // columns of input and the last column is the index of the classification.
  uint32_t TestData(std::string fileName, bool scaled = true);

  std::vector<double> TargetFromDirName(std::string dir);

private:
  // The forward path in a fully connected layer.
  void   ForwardPathFull(Layer* current);

  // The forward path in a convolutional layer.
  void   ForwardPathConv(Layer* current);

  // The forward path in a max pool layer.
  void   ForwadPathMaxpool(Layer* curr_layer);

  // The forward path in a normalisation layer.
  void   ForwardPathLayerNorm(Layer* curr_layer);

  // The backward path in a fully connected layer.
  void   BackwardPathFull(Layer* current);

  // The backward path in a convolutional layer.
  void   BackwardPathConv(Layer* current);

  // The backward path in a max pool layer.
  void   BackwardPathMaxpool(Layer* curr_layer);

  // The backward path in a normalisation layer.
  void   BackwardPathLayerNorm(Layer* curr_layer);

  // The back propagation called that will decide whether 
  // it is FC, Conv or maxpool layer to backward path.
  void   ForwadPath(Layer* current);

  // Train the model on this target values.
  void   TrainData(std::vector<double> values);

  // Get me the next layer (used in forward paths)
  Layer* NextLayer(Layer* curr);

  // Get me the prev layer (used in back propagation paths)
  Layer* PreviousLayer(Layer* curr);

  // giving the index, return me the layer(the layers are numbered sequentially)
  // starting at the input layer as layer 0, and the last layer as (N-1).
  Layer* GetLayer(size_t index);

  // this is used to initialise the layer's weights on start-up (but if the model is found, 
  // the weights gets overwritten with the trained weights)
  double rnd();
  double nrnd();

  // The sigmoid function used in forward path.
  double sigmoid(double x);

  // The sigmoid gradient function that will be used in back propagation path.
  double sigmoid_g(double y);

  // The tanh function used in forward path.
  double tanh(double y);

  // The tanh gradient function that will be used in back propagation path.
  double tanh_g(double y);

  // this is called to calculate the outputs of each neuron according to the 
  // layer's activation function.
  double NonLinearFunction(Layer* layer, double input);

  // this is called to calculate the gradients of each neuron according to the 
  // layer's activation function.
  double FunctionGradient(Layer* layer, double input);

  // This will disable some random neurons during the learning process to avoid
  // overfitting.
  void GenerateDroputNodes(Layer* layer);

private:
  // our list of model layers.
  typedef std::vector<Layer*> Layers;
  Layers layers_;

  // our counter for layer number.
  // this gets incremented every time we create a new layer
  uint32_t layerNum_;

  // the learning rate of our model.
  double learningRate_;

  // the dropout rate of our model used for preventing overfitting.
  double dropout_;

  // The confidence of our last prediction.
  float confidence_;

  // the model name(I created this scheme of model name so as to be able to save/load
  // models according to given names).
  std::string modelName_;

  // A boolean to indicate if our model is currently learning.
  bool training_;

  //A boolean to indicate if we are pruning the model or not.
  bool pruning_;
  // our minimum weight threshold.
  double pruningThreshold_;

  double m_totalError;
};

#endif 
