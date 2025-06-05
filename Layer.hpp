#ifndef __LAYER_HPP__
#define __LAYER_HPP__
/*
   Code Written by : Baher Nicola
   Date            : 12/08/2023
*/

/*
    This is the Neural network layer structure
    This is used by the Model.cpp class to create
    any number of layers to construct a neural network.

    This can be used for image classification.

    The fully connected (FC) layers are tested and functioning properly.
    The convolutional neural network has been tested along with maxpool
    layers and has proven to be working, but it is a bit slow.

    So caution should be taken when using this library's convolutional layers
*/

#include <stdint.h>

class Layer
{
public:

  // Each layer can have its own activation function.
  enum ActivationFunc
  {
    relu,
    leakyRelu,
    sigmoid,
    tanh
  };

  // Each layer can be an input, fully connected, convolutional, maxpool type.
  enum LayerType
  {
    inputLayer,
    convLayer,
    maxpoolLayer,
    fullLayer,
    residualFullLayer,
    dropLayer,
    normalizationLayer
  };


public:
  // Constructor and Destructor
  Layer();
  ~Layer();

public:
  // layer number as each layer in the network
  // will be assigned a number according to its location
  // in the network, so input layer will have the ID 0, 
  // next layer will have the ID 1...etc.
  uint32_t layer_num;

  // Our own type (input, conv, maxpool, full).
  LayerType type;

  // The number of noeurons(perceptrons) for this layer.
  uint32_t n_outputs;

  // the actual outputs of neurons.
  double* outputs;

  bool has_residual_connection;
  int  residual_source_layer;

  // The errors propagated to this layer (used in the back propagation)
  // in our gradient descent method to minimize the errors.
  double* errors;

  // the number of weights coming to this layer
  // from the previous layer
  uint32_t n_weights;

  // The actual layer weights (they are sequentially assigned).
  // So this layer will have w1, w2, w3 ...etc. coming to the first 
  // neuron
  double* weights;

  // The value that we need to update the current weights with
  // to minimize our errors in this layer.
  double* u_weights;

  // The gradient calculated for this layer.
  // It is used in the gradient descent back propagation
  // to calculate the dw (the amount of change needed to update the weight).
  double* gradients;

  // This will be used for Maxpool layers to store which cell in th eprevious layes
  // produced the maximum.
  double* max_indices;

  // The following is used in the normalisation layer
  double* gamma;
  double* beta;
  double* mean;
  double* variance;
  // For storing normalized outputs
  double* normalized_outputs;

  // This is acting like a layer that randomly switch off neurons.
  // We achieve this by multiplying the output of a neuron by 0 or 1 randomly.
  // We generate 0s and 1s randomly based on the drop rate.
  double* dropOut;

  // Our own activation function for this layer.
  ActivationFunc funct;

  // The size of the kernel in convolutional and maxpool layers.
  uint32_t kern_size;

  // The height, width, depth in convolutional and maxpool layers.
  uint32_t height;
  uint32_t width;
  uint32_t depth;

  // The stride, padding used in convolutional and maxpool layers.
  uint32_t stride;
  uint32_t padding;

};

#endif
