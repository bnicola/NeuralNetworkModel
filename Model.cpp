#include "Model.hpp"
#include <iostream>
#ifndef _WIN32
#include <sys/stat.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <ctime>
#endif

//#define DEBUG 

Model::Model(double learning_rate)
  :layerNum_(0),
  learningRate_(learning_rate),
  dropout_(0.2),
  modelName_("Default"),
  training_(false),
  pruning_(false)
{}

Model::~Model()
{}

double Model::rnd()
{
  return ((double)rand() / RAND_MAX);
}

double Model::nrnd()
{
  return (rnd() + rnd() + rnd() + rnd() - 2.0) * 1.724; /* std=1.0 */
}

double Model::sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

double Model::sigmoid_g(double y)
{
  return y * (1.0 - y);
}

double Model::tanh(double y)
{
  return  (exp(y) - exp(-y)) / (exp(y) + exp(-y));
}

double Model::tanh_g(double y)
{
  return  (exp(y) - exp(-y)) / (exp(y) + exp(-y));
}

void Model::SetLearningRate(double rate)
{
  learningRate_ = rate;
}

void Model::SetDropRate(double droprate)
{
  dropout_ = droprate;
}

void Model::SetPruning(bool pruning, double threshold)
{
  pruning_ = pruning;
  pruningThreshold_ = threshold;
  if (pruning_ == true)
    LoadWeights();
}

double Model::NonLinearFunction(Layer* layer, double input)
{
  double result = 0;
  if (layer->funct == Layer::relu)
  {
    result = (input <= 0) ? 0 : input;
  }
  else if (layer->funct == Layer::leakyRelu)
  {
    result = (input <= 0) ? (0.01 * input) : input;
  }
  else if (layer->funct == Layer::sigmoid)
  {
    result = 1 / (1 + exp(-input));
  }
  else if (layer->funct == Layer::tanh)
  {
    result = (exp(input) - exp(-input)) / (exp(input) + exp(-input));
#ifdef _WIN32
    if (isnan(result))
    {
      // Handle NaN case, for example, set result to a specific value
      result = 0.000001;
    }
#endif
  }
  return result;
}

double Model::FunctionGradient(Layer* layer, double input)
{
  double result = 0;
  if (layer->funct == Layer::relu)
  {
    result = (input <= 0) ? 0 : 1;
  }
  else if (layer->funct == Layer::leakyRelu)
  {
    result = (input <= 0) ? 0.01 : 1;
  }
  else if (layer->funct == Layer::sigmoid)
  {
    double sigmoid = 1 / (1 + exp(-input));
    result = sigmoid * (1 - sigmoid);
  }
  else if (layer->funct == Layer::tanh)
  {
    result = 1 - (tanh(input) * tanh(input));
  }

  // Handle NaN case, for example, set result to a specific value
#ifdef _WIN32
  if (isnan(result))
  {
    result = 0.0001;
  }
  else if (isinf(result))
  {
    result = 1.0;
  }
#endif

  return result;
}

void Model::GenerateDroputNodes(Layer* layer)
{
  int numOnes = 0, numZeros = 0;
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (uint32_t i = 0; i < layer->n_outputs; i++)
  {
    // Define the probability of generating 0 or 1, in short the drop rate.
    double probability = dropout_;  // probability of 0, probability of 1

    // Generate random 0s and 1s based on the probability
    int randomValue = (std::rand() < probability * RAND_MAX) ? 0 : 1;
    //std::cout << randomValue << " ";
    layer->dropOut[i] = randomValue;

    /*if (randomValue == 0)
       numZeros++;
    else
       numOnes++;*/
  }
}

//============================================ Public functions ====================================================

Layer* Model::CreateInputLayer(int curr_height, int curr_width, int current_depth)
{
  Layer* inp = new (std::nothrow) Layer();
  if (inp == nullptr)
  {
    return nullptr;
  }

  // if we have no layers created yet, so this must be our first input layer
  if (layers_.size() == 0)
  {
    inp->layer_num = layerNum_++;
    inp->type = Layer::inputLayer;
    inp->width = curr_width;
    inp->height = curr_height;
    inp->depth = current_depth;
    inp->stride = 0;
    inp->padding = 0;
    inp->kern_size = 0;
    inp->funct = Layer::relu;
    inp->n_weights = 0;
    inp->n_outputs = inp->depth * inp->height * inp->width;
    inp->weights = nullptr;
    inp->u_weights = nullptr;

    inp->outputs = new (std::nothrow) double[inp->n_outputs];
    if (inp->outputs == nullptr)
    {
      delete inp;
      return nullptr;
    }

    inp->gradients = nullptr;
    inp->dropOut = nullptr;

    inp->errors = new (std::nothrow) double[inp->n_outputs];
    if (inp->errors == nullptr)
    {
      delete[] inp->outputs;
      delete inp;
      return nullptr;
    }

    // Initialize outputs and errors
    for (int i = 0; i < inp->n_outputs; i++)
    {
      inp->outputs[i] = 0.0;
      inp->errors[i] = 0.0;
    }
  }
  else
  {
    // Handle the case where layers_.size() != 0
    // This might be an error condition, or you might want to handle it differently
    delete inp;
    return nullptr;
  }

  layers_.push_back(inp);
  return inp;
}


Layer* Model::CreateFullLayer(uint32_t numNodes, Layer::ActivationFunc func)
{
  Layer* full = new Layer();
  if (full == nullptr)
  {
    return nullptr;
  }
  else
  {
    // if we have no layers created yet, so this must be our first input layer
    if (layers_.size() == 0)
    {
      full->layer_num = layerNum_++;
      full->type = Layer::inputLayer;
      full->width = numNodes;
      full->height = 1;
      full->depth = 1;
      full->stride = 1;
      full->kern_size = 1;
      full->funct = func;
      full->n_weights = 0;
      full->n_outputs = full->depth * full->height * full->width;
      full->weights = NULL;
      full->u_weights = 0;
      full->outputs = new double[full->n_outputs];
      full->gradients = NULL;
      full->dropOut = NULL;
      full->errors = new double[full->n_outputs];
    }
    else
    {
      full->layer_num = layerNum_++;
      Layer* prev = PreviousLayer(full);
      full->type = Layer::fullLayer;
      full->width = 0;
      full->height = 0;
      full->depth = 0;
      full->stride = 0;
      full->padding = 0;
      full->kern_size = 0;
      full->n_weights = prev->n_outputs * numNodes;
      full->n_outputs = numNodes;

      full->weights = new double[full->n_weights];
      full->u_weights = new double[full->n_weights];
      for (uint32_t i = 0; i < full->n_weights; i++)
      {
        full->weights[i] = 0.1 * nrnd();// random_number;
        full->u_weights[i] = 0.0;// random_number;
      }

      full->outputs = new double[full->n_outputs];
      full->errors = new double[full->n_outputs];
      full->gradients = new double[full->n_outputs];
      full->dropOut = new double[full->n_outputs];
      for (uint32_t i = 0; i < full->n_outputs; i++)
      {
        full->outputs[i] = 0.0;
        full->errors[i] = 0.0;
        full->gradients[i] = 1.0;
        full->dropOut[i] = 1.0;
      }

      full->funct = func;
    }

    layers_.push_back(full);

  }

  return full;
}

Layer* Model::CreateConvLayer(int current_depth, int kernel_size, int stride, int padding, Layer::ActivationFunc funct)
{
  Layer* conv = new Layer();
  if (conv == nullptr)
  {
    return nullptr;
  }
  else
  {
    conv->layer_num = layerNum_++;
    Layer* prev = PreviousLayer(conv);
    conv->type = Layer::convLayer;
    conv->width = ((prev->width - kernel_size) / stride) + 1;
    conv->height = ((prev->height - kernel_size) / stride) + 1;;
    conv->depth = current_depth;
    conv->stride = stride;
    conv->padding = padding;
    conv->kern_size = kernel_size;

    conv->n_weights = prev->depth * kernel_size * kernel_size * current_depth;
    conv->n_outputs = current_depth * conv->height * conv->width;
    conv->errors = new (std::nothrow) double[conv->n_outputs];
    if (conv->errors == nullptr)
    {
      delete conv;
      return nullptr;
    }

    conv->weights = new (std::nothrow) double[conv->n_weights];
    if (conv->weights == nullptr)
    {
      delete[] conv->errors;
      delete conv;
      return nullptr;
    }

    conv->u_weights = new (std::nothrow) double[conv->n_weights];
    if (conv->u_weights == nullptr)
    {
      delete[] conv->errors;
      delete[] conv->weights;
      delete conv;
      return nullptr;
    }

    conv->outputs = new (std::nothrow) double[conv->n_outputs];
    if (conv->outputs == nullptr)
    {
      delete[] conv->errors;
      delete[] conv->weights;
      delete[] conv->u_weights;
      delete conv;
      return nullptr;
    }

    conv->gradients = new (std::nothrow) double[conv->n_outputs];
    if (conv->gradients == nullptr)
    {
      delete[] conv->errors;
      delete[] conv->weights;
      delete[] conv->u_weights;
      delete[] conv->outputs;
      delete conv;
      return nullptr;
    }

    conv->dropOut = new (std::nothrow) double[conv->n_outputs];
    if (conv->dropOut == nullptr)
    {
      delete[] conv->errors;
      delete[] conv->weights;
      delete[] conv->u_weights;
      delete[] conv->outputs;
      delete[] conv->gradients;
      delete conv;
      return nullptr;
    }
    conv->funct = funct;

    for (uint32_t i = 0; i < conv->n_weights; i++)
    {
      // He initialization for ReLU
      double stddev = std::sqrt(2.0 / (kernel_size * kernel_size * prev->depth));
      conv->weights[i] = stddev * nrnd();  // Use your nrnd() function here
      conv->u_weights[i] = 0;
    }

    for (uint32_t i = 0; i < conv->n_outputs; i++)
    {
      conv->outputs[i] = 0.01;  // Small non-zero value
      conv->gradients[i] = 1.0;
      conv->dropOut[i] = 1.0;
      conv->errors[i] = 0.0;
    }

    layers_.push_back(conv);
  }
  return conv;
}

Layer* Model::CreateMaxpoolLayer(uint32_t size, uint32_t stride)
{
  Layer* maxpool = new Layer();
  if (maxpool == nullptr)
  {
    return nullptr;
  }
  else if (layers_.size() == 0)
  {
    return nullptr; // we cannot create a maxpool layer at the beginning
  }
  else
  {
    maxpool->layer_num = layerNum_++;
    Layer* prev = PreviousLayer(maxpool);
    maxpool->type = Layer::maxpoolLayer;
    maxpool->width = prev->width / size;
    maxpool->height = prev->height / size;
    maxpool->depth = prev->depth;
    maxpool->stride = stride;
    maxpool->padding = 0;
    maxpool->kern_size = size;

    maxpool->n_weights = 0;
    maxpool->n_outputs = maxpool->depth * maxpool->height * maxpool->width;
    maxpool->errors = new double[maxpool->n_outputs];
    maxpool->weights = NULL;// new double[maxpool->n_weights];
    maxpool->u_weights = 0;// new double[maxpool->n_weights];
    maxpool->outputs = new double[maxpool->n_outputs];
    maxpool->gradients = new double[maxpool->n_outputs];
    maxpool->dropOut = new double[maxpool->n_outputs];

    for (uint32_t i = 0; i < maxpool->n_outputs; i++)
    {
      maxpool->outputs[i] = 0.0;
      maxpool->errors[i] = 0.0;
      maxpool->gradients[i] = 1.0;
      maxpool->dropOut[i] = 1.0;
    }
  }

  layers_.push_back(maxpool);

  return maxpool;
}

Layer* Model::CreateNormalizationLayer(uint32_t numOutputs)
{
  Layer* layer = new Layer();
  layer->type = Layer::normalizationLayer;
  layer->n_outputs = numOutputs;
  layer->outputs = new double[layer->n_outputs];
  layer->errors = new double[layer->n_outputs];
  layer->gradients = new double[layer->n_outputs];

  // Initialize gamma and beta
  layer->gamma = new double[layer->n_outputs];
  layer->beta = new double[layer->n_outputs];

  layer->n_weights = numOutputs * 2;
  layer->weights = new double[layer->n_weights];
  layer->u_weights = new double[layer->n_weights];

  // For storing mean and variance
  layer->mean = new double;
  layer->variance = new double;

  layer->weights = new double[layer->n_weights];
  layer->u_weights = new double[layer->n_weights];
  for (uint32_t i = 0; i < layer->n_weights; i++)
  {
    layer->weights[i] = 1;// 0.1 * nrnd();// random_number;
    layer->u_weights[i] = 0.0;// random_number;
  }

  // For storing normalized outputs
  layer->normalized_outputs = new double[layer->n_outputs];

  layer->layer_num = layerNum_++;
  layers_.push_back(layer);

  return layer;
}



Layer* Model::NextLayer(Layer* curr)
{
  Layer* next = nullptr;
  int index = curr->layer_num;
  if (index < (layers_.size() - 1))
  {
    next = layers_.at((curr->layer_num) + 1);
  }

  return next;
}

Layer* Model::PreviousLayer(Layer* curr)
{
  Layer* prev = nullptr;
  int index = curr->layer_num;
  if (index > 0)
  {
    prev = layers_.at((curr->layer_num) - 1);
  }

  return prev;
}

Layer* Model::GetLayer(size_t index)
{
  Layer* layer = nullptr;
  if (index < layers_.size())
    layer = layers_.at(index);

  return layer;
}

void Model::ForwadPathFull(Layer* curr_layer)
{
  Layer* prev = PreviousLayer(curr_layer);

  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    curr_layer->outputs[i] = 0;
  }

  int k = 0;
  if (curr_layer != NULL)
  {
    for (uint32_t j = 0; j < curr_layer->n_outputs; j++)
    {
      for (uint32_t i = 0; i < prev->n_outputs; i++)
      {
        curr_layer->outputs[j] += curr_layer->weights[k++] * prev->outputs[i];
      }
      double val = curr_layer->outputs[j];
      if (training_) // allow dropout while training in the forward paths.
      {
        if (curr_layer->layer_num != layers_.size() - 1) //don't produce drop out in the last layer
        {
          GenerateDroputNodes(curr_layer);
        }
        curr_layer->outputs[j] = NonLinearFunction(curr_layer, val) * curr_layer->dropOut[j];
      }
      else
      {
        curr_layer->outputs[j] = NonLinearFunction(curr_layer, val);
      }
      curr_layer->gradients[j] = FunctionGradient(curr_layer, val);
    }
  }
}


void Model::BackwardPathFull(Layer* curr_layer)
{
  if (curr_layer->type != Layer::inputLayer)
  {
    Layer* prev = layers_.at(curr_layer->layer_num - 1);
    if (prev != NULL)
    {
      // Calculate the previous layer errors.
      for (uint32_t i = 0; i < prev->n_outputs; i++)
      {
        prev->errors[i] = 0;
      }
      for (uint32_t i = 0; i < prev->n_outputs; i++)
      {
        int k = 0;
        for (uint32_t j = 0; j < curr_layer->n_outputs; j++)
        {
          prev->errors[i] += curr_layer->weights[k + i] * curr_layer->errors[j] * 0.01/*curr_layer->gradients[j]*/;
          k += prev->n_outputs;
        }
      }

      // calculate delta W (how much each weight should change to enhance the output/decrease the loss function).
      int k = 0;
      for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
      {
        for (uint32_t j = 0; j < prev->n_outputs; j++)
        {
          curr_layer->u_weights[k++] = curr_layer->errors[i] * curr_layer->gradients[i] * prev->outputs[j];
        }
      }

      // now update our weights slightly(using our learning rate).
      for (uint32_t i = 0; i < curr_layer->n_weights; i++)
      {
        curr_layer->weights[i] -= learningRate_ * curr_layer->u_weights[i];
      }
    }
  }
}

void Model::ForwadPathMaxpool(Layer* curr_layer)
{
  Layer* prev_layer = PreviousLayer(curr_layer);
  uint32_t output_height = (prev_layer->height - curr_layer->kern_size) / curr_layer->stride + 1;
  uint32_t output_width = (prev_layer->width - curr_layer->kern_size) / curr_layer->stride + 1;

  int k = 0;

  for (uint32_t z = 0; z < prev_layer->depth; z++)
  {
    for (uint32_t y = 0; y < output_height; y++)
    {
      for (uint32_t x = 0; x < output_width; x++)
      {
        int max_value_index = (z * prev_layer->height * prev_layer->width) + (y * curr_layer->stride * prev_layer->width) + (x * curr_layer->stride);
        double max_value = prev_layer->outputs[max_value_index];// [z] [y * stride] [x * stride] ;
        for (uint32_t k = 0; k < curr_layer->kern_size; k++)
        {
          int y0 = (y * curr_layer->stride) + k;
          for (uint32_t l = 0; l < curr_layer->kern_size; l++)
          {
            int x0 = (x * curr_layer->stride) + l;
            int neighb_pixels = (z * prev_layer->height * prev_layer->width) + (y0 * prev_layer->width) + x0;
            max_value = std::max(max_value, prev_layer->outputs[neighb_pixels]);
          }
        }
        curr_layer->outputs[k++] = max_value;
      }
    }
  }
}

void Model::BackwardPathMaxpool(Layer* curr_layer)
{
  Layer* prev = PreviousLayer(curr_layer);
  int k = 0;
  memset(prev->errors, 0, prev->n_outputs * sizeof(double));

  uint32_t output_height = (prev->height - curr_layer->kern_size) / curr_layer->stride + 1;
  uint32_t output_width = (prev->width - curr_layer->kern_size) / curr_layer->stride + 1;

  //    PrevLayer                      CurrLayer(maxpool)
  // 
  //    n1(0.8)-----------
  //                      |
  //    n2 (0.0)----------
  //                       \___________n(max)(e.g)error = 0.8)
  //    n3(0.0)------------/
  //                      |
  //    n4(0.0)------------
  //
  // the max pool back propagation error propagates its errors to the max node cell that resulted in its value, leaving the rest to 0s.
  //
  // This layer has no weights, so just we need to back propagate the errors.

  for (uint32_t z = 0; z < prev->depth; z++)
  {
    for (uint32_t y = 0; y < output_height/*prev->height - curr_layer->kern_size*/; y++)
    {
      for (uint32_t x = 0; x < output_width/*prev->width - curr_layer->kern_size*/; x++)
      {
        int max_index = (z * prev->height * prev->width) + (y * curr_layer->stride * prev->width) + (x * curr_layer->stride);
        double max_value = prev->outputs[max_index];// previous_output[z][y * curr_layer->stride][x * curr_layer->stride];// assume first node to be the max.
        int max_y = y * curr_layer->stride, max_x = x * curr_layer->stride;

        for (uint32_t m = 0; m < curr_layer->kern_size; m++)
        {
          int y0 = y * curr_layer->stride + m;
          for (uint32_t n = 0; n < curr_layer->kern_size; n++)
          {
            int x0 = x * curr_layer->stride + n;
            int prev_out_index = (z * prev->height * prev->width) + (y0 * prev->width) + x0;
            if (prev->outputs[prev_out_index] >= max_value)
            {
              max_value = prev->outputs[prev_out_index];// previous_output[z][y0][x0];
              max_y = y0;
              max_x = x0;
            }
            if (m == curr_layer->kern_size - 1 && n == (curr_layer->kern_size - 1))
            {
              int prev_index = (z * prev->height * prev->width) + (max_y * prev->width) + max_x;
              prev->errors[prev_index] = curr_layer->errors[k++];
            }
          }
        }
      }
    }
  }
}


void Model::ForwardPathConv(Layer* curr_layer)
{
  Layer* prev_layer = PreviousLayer(curr_layer);
  int currentHeight = prev_layer->height - curr_layer->kern_size + 1;
  int currentWidth = prev_layer->width - curr_layer->kern_size + 1;
  int currentDepth = curr_layer->depth;
  int kernSize = curr_layer->kern_size;

  // Zero out the outputs
  //std::fill(curr_layer->outputs, curr_layer->outputs + curr_layer->n_outputs, 0.0);
  memset(curr_layer->outputs, 0.0, curr_layer->n_outputs);
  for (int ch = 0; ch < currentDepth; ch++)
  {
    for (int yo = 0; yo < currentHeight; yo++)
    {
      for (int xo = 0; xo < currentWidth; xo++)
      {
        double sum = 0.0;
        for (int z = 0; z < prev_layer->depth; z++)
        {
          for (int y = 0; y < kernSize; y++)
          {
            for (int x = 0; x < kernSize; x++)
            {
              int input_idx = (z * prev_layer->height * prev_layer->width) + ((yo + y) * prev_layer->width) + (xo + x);
              int weight_idx = (ch * prev_layer->depth * kernSize * kernSize) + (z * kernSize * kernSize) + (y * kernSize) + x;
              sum += prev_layer->outputs[input_idx] * curr_layer->weights[weight_idx];
            }
          }
        }
        int out_idx = (ch * currentHeight * currentWidth) + (yo * currentWidth) + xo;
        curr_layer->outputs[out_idx] = sum;
      }
    }
  }

  // Apply activation function and calculate gradients
  for (int i = 0; i < curr_layer->n_outputs; i++)
  {
    double val = curr_layer->outputs[i];
    curr_layer->outputs[i] = NonLinearFunction(curr_layer, val);
    curr_layer->gradients[i] = FunctionGradient(curr_layer, val);
  }
}

void Model::BackwardPathConv(Layer* curr_layer)
{
  Layer* prev = PreviousLayer(curr_layer);
  int currentHeight = prev->height - curr_layer->kern_size + 1;
  int currentWidth = prev->width - curr_layer->kern_size + 1;
  int kernSize = curr_layer->kern_size;

  // Zero out the weight gradients
  memset(curr_layer->u_weights, 0, curr_layer->n_weights * sizeof(double));

  // Calculate weight gradients
  for (int zo = 0; zo < curr_layer->depth; zo++)
  {
    for (int z = 0; z < prev->depth; z++)
    {
      for (int y = 0; y < kernSize; y++)
      {
        for (int x = 0; x < kernSize; x++)
        {
          int weightIndex = (zo * prev->depth * kernSize * kernSize) + (z * kernSize * kernSize) + (y * kernSize) + x;
          for (int yo = 0; yo < currentHeight; yo++)
          {
            for (int xo = 0; xo < currentWidth; xo++)
            {
              int outputIndex = (zo * currentHeight * currentWidth) + (yo * currentWidth) + xo;
              int inputIndex = (z * prev->height * prev->width) + ((yo + y) * prev->width) + (xo + x);
              curr_layer->u_weights[weightIndex] += prev->outputs[inputIndex] * curr_layer->errors[outputIndex] * curr_layer->gradients[outputIndex];
            }
          }
        }
      }
    }
  }

  // Zero out previous layer errors
  memset(prev->errors, 0, prev->n_outputs * sizeof(double));

  // Backpropagate errors to previous layer
  for (int z = 0; z < prev->depth; z++)
  {
    for (int y = 0; y < prev->height; y++)
    {
      for (int x = 0; x < prev->width; x++)
      {
        int prevIndex = (z * prev->height * prev->width) + (y * prev->width) + x;
        for (int zo = 0; zo < curr_layer->depth; zo++)
        {
          for (int ky = 0; ky < kernSize; ky++)
          {
            for (int kx = 0; kx < kernSize; kx++)
            {
              int yo = y - ky;
              int xo = x - kx;
              if (yo >= 0 && yo < currentHeight && xo >= 0 && xo < currentWidth)
              {
                int outputIndex = (zo * currentHeight * currentWidth) + (yo * currentWidth) + xo;
                int weightIndex = (zo * prev->depth * kernSize * kernSize) + (z * kernSize * kernSize) + (ky * kernSize) + kx;
                prev->errors[prevIndex] += curr_layer->weights[weightIndex] * curr_layer->errors[outputIndex] * 0.01/*curr_layer->gradients[outputIndex]*/;
              }
            }
          }
        }
      }
    }
  }

  // Update weights
  for (uint32_t i = 0; i < curr_layer->n_weights; i++)
  {
    curr_layer->weights[i] -= learningRate_ * curr_layer->u_weights[i];
  }

  // Gradient clipping (optional, but recommended)
  double max_norm = 1.0;  // Adjust as needed
  double total_norm = 0.0;
  for (uint32_t i = 0; i < curr_layer->n_weights; i++)
  {
    total_norm += curr_layer->u_weights[i] * curr_layer->u_weights[i];
  }
  total_norm = std::sqrt(total_norm);

  if (total_norm > max_norm)
  {
    double scale = max_norm / total_norm;
    for (uint32_t i = 0; i < curr_layer->n_weights; i++)
    {
      curr_layer->u_weights[i] *= scale;
    }
  }
}

void Model::ForwardPathLayerNorm(Layer* curr_layer)
{
  Layer* prev = PreviousLayer(curr_layer);

  // Calculate mean
  double mean = 0.0;
  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    mean += prev->outputs[i];
  }
  mean /= curr_layer->n_outputs;

  // Calculate variance
  double variance = 0.0;
  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    variance += (prev->outputs[i] - mean) * (prev->outputs[i] - mean);
  }
  variance /= curr_layer->n_outputs;

  // Normalize and scale
  double epsilon = 1e-5;  // Small value to avoid division by zero
  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    double normalized = (prev->outputs[i] - mean) / sqrt(variance + epsilon);
    curr_layer->outputs[i] = curr_layer->weights[i] * normalized + curr_layer->weights[i + curr_layer->n_outputs];
  }
}


void Model::BackwardPathLayerNorm(Layer* curr_layer)
{
  Layer* prev = PreviousLayer(curr_layer);

  // Calculate mean and variance (same as in forward pass)
  double mean = 0.0;
  double variance = 0.0;
  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    mean += prev->outputs[i];
  }
  mean /= curr_layer->n_outputs;

  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    variance += (prev->outputs[i] - mean) * (prev->outputs[i] - mean);
  }
  variance /= curr_layer->n_outputs;

  double epsilon = 1e-5;
  double std_dev = sqrt(variance + epsilon);

  // Calculate gradients for gamma and beta
  //                                 X1-u
  // X1 ----------------> gamma * (---------) + B
  //                                 v^0.5
  //
  // the Loss is:
  //                                          X1-u
  // Loss = (T-O)^2 -------> (T -  gamma * (---------) + B)^2
  //                                           v^0.5
  //
  //
  // dLoss/dgamma = 2 * Error * (X1-u/v^0.5) -------> error[i] * normalised[i]
  // dLoss/dbeta  = 2 * Error * (1)          -------> error[i]

  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    double normalized = (prev->outputs[i] - mean) / std_dev;
    curr_layer->u_weights[i] = curr_layer->errors[i] * normalized;  // dL/dgamma
    curr_layer->u_weights[i + curr_layer->n_outputs] = curr_layer->errors[i];  // dL/dbeta
  }

  // Calculate gradients for previous layer
  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    double dx_normalized = curr_layer->errors[i] * curr_layer->weights[i];
    double dvariance = -0.5 * dx_normalized * (prev->outputs[i] - mean) * pow(variance + epsilon, -1.5);
    double dmean = -dx_normalized / std_dev;

    prev->errors[i] = dx_normalized / std_dev +
      dvariance * 2 * (prev->outputs[i] - mean) / curr_layer->n_outputs +
      dmean / curr_layer->n_outputs;
  }

  // Simplified error propagation
  //for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  //{
  //  prev->errors[i] = curr_layer->errors[i] * curr_layer->weights[i];  // gamma is stored in weights[i]
  //}

  // Update weights
  for (uint32_t i = 0; i < 2 * curr_layer->n_outputs; i++)
  {
    curr_layer->weights[i] -= learningRate_ * curr_layer->u_weights[i];
  }
}

void Model::ForwadPath(Layer* layer)
{
  if (layer->type == Layer::fullLayer)
  {
    ForwadPathFull(layer);
  }
  else if (layer->type == Layer::convLayer)
  {
    ForwardPathConv(layer);
  }
  else if (layer->type == Layer::maxpoolLayer)
  {
    ForwadPathMaxpool(layer);
  }
  else if (layer->type == Layer::normalizationLayer)
  {
    ForwardPathLayerNorm(layer);
  }
}

void Model::TrainData(std::vector<double> values)
{

  Layer* curr = GetLayer(layers_.size() - 1);
  Layer* prev = PreviousLayer(curr);

  for (uint32_t i = 0; i < curr->n_outputs; i++)
  {
    double outVal = curr->outputs[i];
    double error = curr->outputs[i] - values[i];
    curr->errors[i] = (curr->outputs[i] - values[i]);
  }

  while (curr != nullptr)
  {
    if (curr->type == Layer::convLayer /*&& prev->type == Layer::convLayer*/)
    {
      BackwardPathConv(curr);
    }
    else if (curr->type == Layer::maxpoolLayer)
    {
      BackwardPathMaxpool(curr);
    }
    else if (curr->type == Layer::normalizationLayer)
    {
      BackwardPathLayerNorm(curr);
    }
    else
    {
      BackwardPathFull(curr);
    }

    curr = PreviousLayer(curr);
  }
}

void Model::InputData(std::vector<double> inputs)
{
  Layer* input = GetLayer(0);;
  int index = 0;
  for (uint32_t i = 0; i < input->depth; i++)
  {
    for (uint32_t j = 0; j < input->height; j++)
    {
      for (uint32_t k = 0; k < input->width; k++)
      {
        input->outputs[index] = inputs[index];
        index++;
      }
    }
  }

  Layer* curr = NextLayer(input);
  while (curr != NULL)
  {
    ForwadPath(curr);
    curr = NextLayer(curr);
  }
}

std::vector<double> Model::GetOutput()
{
  Layer* layer = GetLayer((layers_.size() - 1));
  std::vector<double> outputs;
  for (uint32_t i = 0; i < layer->n_outputs; i++)
  {
    double value = layer->outputs[i];
    outputs.push_back(layer->outputs[i]);
  }

  return outputs;
}

bool Model::SaveModel(std::string Model)
{
  modelName_ = Model;
  std::string modelPath = Model + "/Model.txt";
  struct stat info;
  if (stat(modelPath.c_str(), &info) != 0)
  {
    if (errno == ENOENT)
    {
      std::string command = "mkdir " + Model;
      system(command.c_str());
    }
  }

  FILE* file = fopen(modelPath.c_str(), "w"); // Open the file in write mode
  if (file != NULL)
  {
    fprintf(file, "NumberOfLayers = %zd\n", layers_.size());
    for (uint32_t i = 0; i < layers_.size(); i++)
    {
      std::string type = (layers_.at(i)->type == Layer::fullLayer) ? "FullLayer" : (layers_.at(i)->type == Layer::inputLayer) ? "InputLayer" : (layers_.at(i)->type == Layer::maxpoolLayer) ? "MaxpoolLayer" : "ConvLayer";
      std::string func = (layers_.at(i)->funct == Layer::sigmoid) ? "sigmoid" : (layers_.at(i)->funct == Layer::tanh) ? "tanh" : (layers_.at(i)->funct == Layer::leakyRelu) ? "leakyrelu" : "relu";
      if (type == "ConvLayer" || type == "InputLayer")
      {
        fprintf(file, "Layer%d_Nodes = %d, type = %s, function = %s, height = %d, width = %d, depth = %d, kern_size = %d, stride = %d, padding = %d\n", i, layers_.at(i)->n_outputs, type.c_str(), func.c_str(), layers_.at(i)->height, layers_.at(i)->width, layers_.at(i)->depth, layers_.at(i)->kern_size, layers_.at(i)->stride, layers_.at(i)->padding);
      }
      else if (type == "FullLayer")
      {
        fprintf(file, "Layer%d_Nodes = %d, type = %s, function = %s\n", i, layers_.at(i)->n_outputs, type.c_str(), func.c_str());
      }
      else if (type == "MaxpoolLayer")
      {
        fprintf(file, "Layer%d_Nodes = %d, type = %s, kern_size = %d, stride = %d\n", i, layers_.at(i)->n_outputs, type.c_str(), layers_.at(i)->kern_size, layers_.at(i)->stride);
      }
      else if (type == "NormalisationLayer")
      {
        fprintf(file, "Layer%d_Nodes = %d, type = %s\n", i, layers_.at(i)->n_outputs, type.c_str());
        //fprintf(file, "Layer%d_Nodes = %d, type = %s, kern_size = %d, stride = %d\n", i, layers_.at(i)->n_outputs, type.c_str(), layers_.at(i)->kern_size, layers_.at(i)->stride);
      }
    }
    fclose(file);
  }
  else
  {
    printf("Error opening the file for writing\n");
  }

  return LoadWeights();
}

void Model::SaveWeights(std::string Model)
{
  if (Model == "" || Model == "Default")
  {
    Model = modelName_;
  }
  Layer* current = layers_.at(0);
  while (current != NULL)
  {
    int id = current->layer_num;
    char filename[20];
    sprintf(filename, "Layer_%d.txt", current->layer_num);
    std::string layerWeights = Model + "/" + std::string(filename);
    FILE* fp;
    fp = fopen(layerWeights.c_str(), "w");
    if (fp)
    {
      for (int i = 0; i < current->n_weights; i++)
      {
        float weight = current->weights[i];
        fprintf(fp, "%g\n", current->weights[i]);
      }

      fclose(fp);
    }
    current = NextLayer(current);
  }

  if (pruning_ == true)
  {
    current = layers_.at(0);

    while (current != NULL)
    {
      int id = current->layer_num;
      char filename[20];
      sprintf(filename, "PrunedLayer_%d.txt", current->layer_num);
      std::string layerWeights = Model + "/" + std::string(filename);
      FILE* fp;
      fp = fopen(layerWeights.c_str(), "w");
      Layer* prev_layer = PreviousLayer(current);

      if (current->type == Layer::fullLayer)
      {
        if (prev_layer != nullptr)
        {
          int prev_layer_outputs = prev_layer->n_outputs;
          int index = 0;
          for (int i = 0; i < current->n_outputs; i++)
          {
            int prevIndex = 0;
            for (int j = 0; j < prev_layer->n_outputs; j++)
            {
              double rounded_weight = std::round(current->weights[index] * 1000.0) / 1000.0;
              //printf("%f\n", current->weights[index]);
              if (current->weights[index] != 0)
              {
                std::string output = "Layer_" + std::to_string(current->layer_num) + "  Previous Node[" + std::to_string(j) + "] ---> " + "Node[" + std::to_string(i) + "], " + ", weight[" + std::to_string(current->weights[index]) + "]\n";
#ifdef _WIN32
                fprintf(fp, output.c_str());
#else
                fputs(output.c_str(), fp);
#endif
              }
              index++;
              prevIndex++;
            }
            fprintf(fp, "\n");
          }
        }
      }

      fclose(fp);
      current = NextLayer(current);
    }
  }
}

bool Model::LoadModel(std::string Model)
{
  layers_.clear();
  layerNum_ = 0;
  bool modelExist = false;
  if (Model != "")
  {
    modelName_ = Model;
  }
  std::string modelPath = Model + "/Model.txt";
  FILE* file = fopen(modelPath.c_str(), "r"); // Open the file in read mode
  if (file != NULL)
  {
    modelExist = true;
    char line[300];
    uint32_t numLayers, layerNodes[40], layerHeight[40], layerWidth[40], layerDepth[40], layerKernSize[40], layerStride[40], layerPadding[40];
    char stype[40];
    char sfunction[40];
    std::vector<std::string> types;
    std::vector<std::string> functions;
    int k = 0;
    while (fgets(line, sizeof(line), file))
    {
      if (sscanf(line, "NumberOfLayers = %d\n", &numLayers) == 1)
      {
        //printf("Number of Layers: %d\n", numLayers);
      }
      else if (sscanf(line, "Layer%d_Nodes = %d, type =  %19[^,], function = %19[^,], height = %d, width = %d, depth = %d, kern_size = %d, stride = %d, padding = %d\n", &k, &layerNodes[k], stype, sfunction, &layerHeight[k], &layerWidth[k], &layerDepth[k], &layerKernSize[k], &layerStride[k], &layerPadding[k]) == 10)
      {
        types.push_back(stype);
        functions.push_back(sfunction);
        k++;
      }
      else if (sscanf(line, "Layer%d_Nodes = %d, type = %19[^,], function = %19[^\n]", &k, &layerNodes[k], stype, sfunction) == 4)
      {
        types.push_back(stype);
        functions.push_back(sfunction);
        k++;
      }
      else if (sscanf(line, "Layer%d_Nodes = %: 12pt; color: rgb(0, 0, 0);">
    uint32_t columns = 0;
    std::vector<double> ins;
    std::vector<double> outs;
    while (std::getline(ss, value, ','))
    {
      values.push_back(std::stod(value));
      columns++;
    }
    for (uint32_t i = 0; i < columns - 1; i++)
    {
      values[i] = (values[i] == 0) ? 0.001 : values[i];
      values[i] = (scaled) ? values[i] / 255 : values[i];
      values[i] = (values[i] == 1) ? 0.999 : values[i];
      ins.push_back(values[i]);
    }
    for (uint32_t i = columns - 1; i < columns; i++)
    {
      outputs.push_back(values[i]);
    }
    inputs.push_back(ins);
  }

  file.close();
}

void Model::TrainDataSet(std::string fileName, uint32_t epochs, uint32_t batchSize, double minLearningRate, bool scaled)
{
  std::vector<std::vector<double>> ins;
  std::vector<uint32_t> outs;
  ReadCsvFile(fileName, ins, outs, scaled);
  srand(time(NULL));
  uint32_t classes = GetLayer(layers_.size() - 1)->n_outputs;
  double initialLr = learningRate_;
  double minimumLr = minLearningRate;
  double learningSlope = (minimumLr - initialLr) / epochs;
  for (uint32_t i = 0; i < epochs; i++)
  {
    learningRate_ = (learningSlope * i) + initialLr;// pow(0.999, i)* learningRate_;
    //if (i%100 == 0)
    printf("Training in epoch %d\n", i);
    for (uint32_t j = 0; j < outs.size(); j++)
    {
      InputData(ins.at(j));
      std::vector<double> target_out;
      for (uint32_t c = 0; c < classes; c++)
      {
        target_out.push_back(0);
      }
      for (uint32_t c = 0; c < classes; c++)
      {
        if (c == outs[j])
        {
          target_out.at(c) = 1;
        }
      }
      TrainData(target_out);


      if (j % batchSize == 0)
      {
        SaveWeights(modelName_);
      }
    }
  }
  SaveWeights(modelName_);
}

uint32_t Model::GetMax(std::vector<double> values)
{
  uint32_t ind = 0;
  double prev = values[0];
  for (uint32_t j = 0; j < values.size(); j++)
  {
    if (values.at(j) >= prev)
    {
      ind = j;
      prev = values.at(j);
    }
  }

  return ind;
}

void Model::LearnData(std::vector<double> inputs, std::vector<double> outputs)
{
  training_ = true;
  InputData(inputs);
  TrainData(outputs);
  double errors = GetTotalError();
  //printf("Errors = %g\n", errors);
  //double errors = GetTotalError(layers_.at(layers_.size() - 1));
  training_ = false;
}

double Model::GetTotalError()
{
  double total = 0;
  Layer* lastLayer = layers_.at(layers_.size() - 1);
  for (uint32_t i = 0; i < lastLayer->n_outputs; i++)
  {
    double e = lastLayer->errors[i];
    total += e * e;
  }
  total = total / lastLayer->n_outputs;
  return total;
}

uint32_t Model::TestData(std::string fileName, bool scaled)
{
  int correct = 0;
  std::vector<std::vector<double>> ins;
  std::vector<uint32_t> outs;
  ReadCsvFile(fileName, ins, outs, scaled);
  srand(time(NULL));
  uint32_t classes = GetLayer(layers_.size() - 1)->n_outputs;

  for (int i = 0; i < ins.size(); i++)
  {
    InputData(ins.at(i));
    std::vector<double> predictions = GetOutput();
    double prev = predictions.at(0);
    uint32_t ind = 0;
    for (uint32_t j = 0; j < classes; j++)
    {
      if (predictions.at(j) > prev)
      {
        ind = j;
        prev = predictions.at(j);
      }
    }

    if (ind == outs.at(i))
    {
      correct++;
    }
  }

  return correct;
}

void  Model::AddClass(uint32_t index, std::string label)
{
  Classify newclass;
  newclass.index = index;
  newclass.label = label;
  classes_.push_back(newclass);
}

uint32_t Model::FindClassFromDir(std::string dir)
{
  bool found = false;
  uint32_t symbolIndex = 0;
  for (int i = 0; i < classes_.size() && (found == false); i++)
  {
    if (classes_.at(i).label == dir)
    {
      found = true;
      symbolIndex = i;
    }
  }

  return symbolIndex;

}

float Model::GetCurrentConfidence()
{
  return confidence_;
}

std::string Model::ClassifyInput(std::vector<double> inputs)
{
  std::string classify = "";
  std::vector<double> output;

  InputData(inputs);
  output = GetOutput();

  uint32_t symbolIndex = GetMax(output);
  confidence_ = output.at(symbolIndex) * 100; // how confident we are about this prediction
  classify = classes_.at(symbolIndex).label;

  return classify;
}


std::vector<double> Model::TargetFromDirName(std::string dir)
{
  bool found = false;
  std::vector<double> target;
  target.resize(classes_.size());

  for (int i = 0; i < classes_.size() && (found == false); i++)
  {
    if (classes_.at(i).label == dir)
    {
      target.at(i) = 0.99999;
    }
    else
    {
      target.at(i) = 0.0001;
    }
  }

  return target;
}

