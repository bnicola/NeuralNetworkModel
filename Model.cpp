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

Model::Model(double learning_rate, int batch_size)
  :layerNum_(0),
  learningRate_(learning_rate),
  dropout_(0.0),
  modelName_("Default"),
  training_(false),
  pruning_(false),
  batchSize_(batch_size),
  currentBatchCount_(0),
  momentum_(0.9)
{
}

//---------------------------------------------------------------------------------------------------------------

Model::~Model()
{
}

//---------------------------------------------------------------------------------------------------------------

double Model::rnd()
{
  return ((double)rand() / RAND_MAX);
}

//---------------------------------------------------------------------------------------------------------------

double Model::nrnd()
{
  return (rnd() + rnd() + rnd() + rnd() - 2.0) * 1.724; /* std=1.0 */
}

//---------------------------------------------------------------------------------------------------------------

double Model::sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

//---------------------------------------------------------------------------------------------------------------

double Model::sigmoid_g(double y)
{
  return y * (1.0 - y);
}

//---------------------------------------------------------------------------------------------------------------

double Model::tanh(double y)
{
  return  (exp(y) - exp(-y)) / (exp(y) + exp(-y));
}

//---------------------------------------------------------------------------------------------------------------

double Model::tanh_g(double y)
{
  double tanh_val = tanh(y);
  return 1.0 - tanh_val * tanh_val;
}

//---------------------------------------------------------------------------------------------------------------

void Model::SetLearningRate(double rate)
{
  learningRate_ = rate;
}

//---------------------------------------------------------------------------------------------------------------

double Model::GetLearningRate()
{
  return learningRate_;
}
//---------------------------------------------------------------------------------------------------------------
void Model::SetMomentum(double momentum)
{
  momentum_ = momentum;
}

//---------------------------------------------------------------------------------------------------------------

double Model::GetMomrntum()
{
  return momentum_;
}

//---------------------------------------------------------------------------------------------------------------

void Model::SetDropRate(double droprate)
{
  dropout_ = droprate;
}

//---------------------------------------------------------------------------------------------------------------

void Model::SetBatchSize(int batch_size)
{
  batchSize_ = batch_size;
  currentBatchCount_ = 0;
}

//---------------------------------------------------------------------------------------------------------------

void Model::ResetBatch()
{
  currentBatchCount_ = 0;
  // Reset accumulated gradients
  for (auto& layer : layers_) 
  {
    if (layer->u_weights != nullptr) 
    {
      for (uint32_t i = 0; i < layer->n_weights; i++) 
      {
        layer->u_weights[i] = 0.0;
      }
    }
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::SetPruning(bool pruning, double threshold)
{
  pruning_ = pruning;
  pruningThreshold_ = threshold;
  if (pruning_ == true)
    LoadWeights();
}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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
    layer->dropOut[i] = randomValue/(1- dropout_);
  }
}

//---------------------------------------------------------------------------------------------------------------

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
    inp->has_residual_connection = false;
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

//---------------------------------------------------------------------------------------------------------------

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
      full->has_residual_connection = false;
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
      full->has_residual_connection = false;
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
      full->velocities = new double[full->n_weights];
      for (uint32_t i = 0; i < full->n_weights; i++)
      {
        full->velocities[i] = 0.0;
      }
      srand(42);
      for (uint32_t i = 0; i < full->n_weights; i++)
      {
        double fan_in = prev->n_outputs;
        double fan_out = numNodes;
        double xavier_std = sqrt(2.0 / (fan_in + fan_out));
        full->weights[i] = xavier_std * nrnd();
        full->u_weights[i] = 0.0;
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

//---------------------------------------------------------------------------------------------------------------

Layer* Model::CreateSoftmaxLayer()
{
  Layer* softmax_layer = new Layer();
  Layer* prev = GetLayer(layers_.size() - 1);

  softmax_layer->layer_num = layerNum_++;
  softmax_layer->type = Layer::softmax;
  softmax_layer->n_outputs = prev->n_outputs; // Must match previous layer

  // No weights, biases, or gradients needed for this layer itself
  softmax_layer->n_weights = 0;
  softmax_layer->weights = nullptr;
  softmax_layer->u_weights = nullptr;
  softmax_layer->gradients = nullptr;

  softmax_layer->outputs = new double[softmax_layer->n_outputs];
  softmax_layer->errors = new double[softmax_layer->n_outputs];

  layers_.push_back(softmax_layer);
  return softmax_layer;
}

//---------------------------------------------------------------------------------------------------------------

Layer* Model::CreateAttentionLayer(uint32_t seq_length, uint32_t d_model, uint32_t d_k, uint32_t d_v, uint32_t num_heads)
{
  // Default dimensions if not specified
  if (d_k == 0) d_k = d_model;
  if (d_v == 0) d_v = d_model;

  Layer* layer = new Layer();
  layer->type = Layer::attentionLayer;
  layer->layer_num = layerNum_++;

  Layer* prev = PreviousLayer(layer);
  // Input should be seq_length * d_model (flattened sequence)
  //layer->n_inputs  = seq_length * d_model;
  layer->n_outputs = seq_length * d_model;  // Same size output

  // Initialize attention configuration
  layer->attention_config             = new Layer::AttentionConfig();
  layer->attention_config->seq_length = seq_length;
  layer->attention_config->d_model    = d_model;
  layer->attention_config->d_k        = d_k;
  layer->attention_config->d_v        = d_v;
  layer->attention_config->num_heads  = num_heads;
  layer->attention_config->use_multi_head = (num_heads > 1);

  // Allocate memory
  layer->outputs   = new double[layer->n_outputs];
  layer->errors    = new double[layer->n_outputs];
  layer->gradients = new double[layer->n_outputs];

  // Weight matrices: [d_model x d_k/d_v] for each projection
  uint32_t wq_size = d_model * d_k * num_heads;
  uint32_t wk_size = d_model * d_k * num_heads;
  uint32_t wv_size = d_model * d_v * num_heads;
  uint32_t wo_size = d_v * d_model * num_heads;

  layer->query_weights = new double[wq_size];
  layer->key_weights = new double[wk_size];
  layer->value_weights = new double[wv_size];
  layer->output_weights = new double[wo_size];

  layer->query_gradients = new double[wq_size];
  layer->key_gradients = new double[wk_size];
  layer->value_gradients = new double[wv_size];
  layer->output_gradients = new double[wo_size];

  // Temporary storage
  layer->temp_q = new double[seq_length * d_k * num_heads];
  layer->temp_k = new double[seq_length * d_k * num_heads];
  layer->temp_v = new double[seq_length * d_v * num_heads];

  // Attention matrices
  layer->attention_scores = new double[seq_length * seq_length * num_heads];
  layer->attention_weights = new double[seq_length * seq_length * num_heads];

  // Initialize weights with Xavier initialization
  double xavier_std = sqrt(2.0 / (d_model + d_k));
  for (uint32_t i = 0; i < wq_size; i++)
  {
    layer->query_weights[i] = xavier_std * nrnd();
    layer->query_gradients[i] = 0.0;
  }
  for (uint32_t i = 0; i < wk_size; i++)
  {
    layer->key_weights[i] = xavier_std * nrnd();
    layer->key_gradients[i] = 0.0;
  }
  for (uint32_t i = 0; i < wv_size; i++) 
  {
    layer->value_weights[i] = xavier_std * nrnd();
    layer->value_gradients[i] = 0.0;
  }
  for (uint32_t i = 0; i < wo_size; i++) 
  {
    layer->output_weights[i] = xavier_std * nrnd();
    layer->output_gradients[i] = 0.0;
  }

  layers_.push_back(layer);

  return layer;
}

//---------------------------------------------------------------------------------------------------------------

Layer* Model::CreateResidualFullLayer(uint32_t numNodes, Layer::ActivationFunc func, int residual_source_layer)
{
  Layer* full = CreateFullLayer(numNodes, func);
  //full->type = Layer::residualFullLayer;
  if (full != nullptr)
  {
    // Set residual connection properties
    full->has_residual_connection = true;
    full->residual_source_layer = residual_source_layer;

    // Ensure the dimensions match for the residual connection
    // If residual_source_layer is -1, we use the previous layer
    Layer* source_layer;
    if (residual_source_layer == -1)
    {
      source_layer = PreviousLayer(full);
    }
    else
    {
      source_layer = GetLayer(residual_source_layer);
    }

    // Check if dimensions match for the residual connection
    if (source_layer->n_outputs != full->n_outputs)
    {
      // If dimensions don't match, we need a projection matrix
      // This would require additional weights to transform the input dimension
      // to match the output dimension
      //full->projection_needed = true;
      //full->projection_weights = new double[source_layer->n_outputs * full->n_outputs];
      //full->projection_u_weights = new double[source_layer->n_outputs * full->n_outputs];

      //// Initialize projection weights
      //for (uint32_t i = 0; i < source_layer->n_outputs * full->n_outputs; i++)
      //{
      //  full->projection_weights[i] = 0.1 * nrnd();
      //  full->projection_u_weights[i] = 0.0;
      //}
    }
  }
  return full;
}

//---------------------------------------------------------------------------------------------------------------

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
      conv->weights[i] = stddev * nrnd();
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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

Layer* Model::CreateNormalizationLayer(/*uint32_t numOutputs*/)
{
  if (layers_.size() > 0) // we cannot have a normalisation layer as input layer
  {
    Layer* layer = new Layer();
    layer->type = Layer::normalizationLayer;
    layer->layer_num = layerNum_++;
    Layer* prev = PreviousLayer(layer);
    layer->n_outputs = prev->n_outputs;
    layer->outputs = new double[layer->n_outputs];
    layer->errors = new double[layer->n_outputs];
    layer->gradients = new double[layer->n_outputs];

    // Initialize gamma and beta
    layer->gamma = new double[layer->n_outputs];
    layer->beta = new double[layer->n_outputs];

    layer->n_weights = prev->n_outputs * 2;
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

    layers_.push_back(layer);

    return layer;
  }
}

//---------------------------------------------------------------------------------------------------------------

#include "Model.hpp"
#include <vector>
#include <iostream>

// Add this to your Layer class in the header file
struct RNNConfig {
  uint32_t sequence_length;
  uint32_t hidden_size;
  uint32_t input_size;
};

// Add these members to your Layer class:
// RNNConfig* rnn_config;
// double* hidden_states;     // Store all hidden states [seq_len x hidden_size]
// double* input_weights;     // Wxh weights [input_size x hidden_size]
// double* hidden_weights;    // Whh weights [hidden_size x hidden_size]
// double* bias_weights;      // bias [hidden_size]
// double* prev_hidden;       // Previous hidden state [hidden_size]

//---------------------------------------------------------------------------------------------------------------
//
//Layer* Model::CreateRNNLayer(uint32_t sequence_length, uint32_t hidden_size)
//{
//  Layer* rnn = new Layer();
//  if (rnn == nullptr) return nullptr;
//
//  rnn->layer_num = layerNum_++;
//  rnn->type = Layer::rnnLayer; // Add this enum value to your Layer::LayerType
//  rnn->has_residual_connection = false;
//
//  Layer* prev = PreviousLayer(rnn);
//  if (prev == nullptr) 
//  {
//    delete rnn;
//    return nullptr;
//  }
//
//  // Calculate input size per timestep (prev layer output / sequence length)
//  uint32_t input_size = prev->n_outputs / sequence_length;
//
//  // Setup RNN configuration
//  rnn->rnn_config = new RNNConfig();
//  rnn->rnn_config->sequence_length = sequence_length;
//  rnn->rnn_config->hidden_size = hidden_size;
//  rnn->rnn_config->input_size = input_size;
//
//  // Output: hidden state at each timestep
//  rnn->n_outputs = sequence_length * hidden_size;
//
//  // Weight sizes
//  uint32_t wxh_size = input_size * hidden_size;   // Input to hidden
//  uint32_t whh_size = hidden_size * hidden_size;  // Hidden to hidden
//  uint32_t bias_size = hidden_size;               // Bias
//
//  rnn->n_weights = wxh_size + whh_size + bias_size;
//
//  // Allocate memory
//  rnn->outputs = new double[rnn->n_outputs];
//  rnn->errors = new double[rnn->n_outputs];
//  rnn->gradients = new double[rnn->n_outputs];
//
//  // RNN specific memory
//  rnn->hidden_states = new double[sequence_length * hidden_size];
//  rnn->input_weights = new double[wxh_size];
//  rnn->hidden_weights = new double[whh_size];
//  rnn->bias_weights = new double[bias_size];
//  rnn->prev_hidden = new double[hidden_size];
//
//  // Weight gradients
//  rnn->weights = new double[rnn->n_weights];
//  rnn->u_weights = new double[rnn->n_weights];
//
//  // Initialize weights with Xavier initialization
//  double xavier_input = sqrt(2.0 / (input_size + hidden_size));
//  double xavier_hidden = sqrt(2.0 / (hidden_size + hidden_size));
//
//  // Initialize input weights (Wxh)
//  for (uint32_t i = 0; i < wxh_size; i++) 
//  {
//    rnn->input_weights[i] = xavier_input * nrnd();
//  }
//
//  // Initialize hidden weights (Whh)  
//  for (uint32_t i = 0; i < whh_size; i++) 
//  {
//    rnn->hidden_weights[i] = xavier_hidden * nrnd();
//  }
//
//  // Initialize bias weights
//  for (uint32_t i = 0; i < bias_size; i++)
//  {
//    rnn->bias_weights[i] = 0.0; // Biases typically start at zero
//  }
//
//  // Initialize states
//  for (uint32_t i = 0; i < rnn->n_outputs; i++) {
//    rnn->outputs[i] = 0.0;
//    rnn->errors[i] = 0.0;
//    rnn->gradients[i] = 0.0;
//  }
//
//  for (uint32_t i = 0; i < hidden_size; i++) {
//    rnn->prev_hidden[i] = 0.0;
//  }
//
//  rnn->funct = Layer::tanh; // RNNs typically use tanh activation
//
//  layers_.push_back(rnn);
//  return rnn;
//}

//---------------------------------------------------------------------------------------------------------------

//void Model::ForwardPathRNN(Layer* curr_layer)
//{
//  Layer* prev = PreviousLayer(curr_layer);
//  RNNConfig* config = curr_layer->rnn_config;
//
//  // Clear previous hidden state at start of sequence
//  for (uint32_t i = 0; i < config->hidden_size; i++)
//  {
//    curr_layer->prev_hidden[i] = 0.0;
//  }
//
//  // Process each timestep
//  for (uint32_t t = 0; t < config->sequence_length; t++)
//  {
//    // Get input for this timestep
//    double* input_t = &prev->outputs[t * config->input_size];
//    double* hidden_t = &curr_layer->hidden_states[t * config->hidden_size];
//
//    // Compute hidden state: h_t = tanh(Wxh * x_t + Whh * h_{t-1} + b)
//    for (uint32_t h = 0; h < config->hidden_size; h++)
//    {
//      double sum = curr_layer->bias_weights[h];
//
//      // Input contribution: Wxh * x_t
//      for (uint32_t i = 0; i < config->input_size; i++)
//      {
//        sum += curr_layer->input_weights[i * config->hidden_size + h] * input_t[i];
//      }
//
//      // Hidden contribution: Whh * h_{t-1}  
//      for (uint32_t prev_h = 0; prev_h < config->hidden_size; prev_h++)
//      {
//        sum += curr_layer->hidden_weights[prev_h * config->hidden_size + h] * curr_layer->prev_hidden[prev_h];
//      }
//
//      // Apply activation function (tanh)
//      hidden_t[h] = NonLinearFunction(curr_layer, sum);
//
//      // Store in outputs (flattened)
//      curr_layer->outputs[t * config->hidden_size + h] = hidden_t[h];
//    }
//
//    // Update previous hidden state for next timestep
//    for (uint32_t h = 0; h < config->hidden_size; h++) 
//    {
//      curr_layer->prev_hidden[h] = hidden_t[h];
//    }
//  }
//}
//
////---------------------------------------------------------------------------------------------------------------
//
//void Model::BackwardPathRNN(Layer* curr_layer)
//{
//  Layer* prev = PreviousLayer(curr_layer);
//  RNNConfig* config = curr_layer->rnn_config;
//
//  // Clear previous layer errors
//  for (uint32_t i = 0; i < prev->n_outputs; i++) {
//    prev->errors[i] = 0.0;
//  }
//
//  // Initialize gradient accumulators
//  std::vector<double> wxh_grad(config->input_size * config->hidden_size, 0.0);
//  std::vector<double> whh_grad(config->hidden_size * config->hidden_size, 0.0);
//  std::vector<double> bias_grad(config->hidden_size, 0.0);
//
//  // Hidden state gradient for next timestep (starts at 0)
//  std::vector<double> next_h_grad(config->hidden_size, 0.0);
//
//  // Backward through time (from last timestep to first)
//  for (int t = config->sequence_length - 1; t >= 0; t--)
//  {
//    // Get current timestep data
//    double* input_t = &prev->outputs[t * config->input_size];
//    double* hidden_t = &curr_layer->hidden_states[t * config->hidden_size];
//    double* h_prev = (t == 0) ? nullptr : &curr_layer->hidden_states[(t - 1) * config->hidden_size];
//
//    // Current hidden gradient = output error + gradient from next timestep
//    std::vector<double> h_grad(config->hidden_size);
//    for (uint32_t h = 0; h < config->hidden_size; h++) 
//    {
//      h_grad[h] = curr_layer->errors[t * config->hidden_size + h] + next_h_grad[h];
//    }
//
//    // Compute raw hidden state (before activation)
//    std::vector<double> raw_hidden(config->hidden_size);
//    for (uint32_t h = 0; h < config->hidden_size; h++) 
//    {
//      double sum = curr_layer->bias_weights[h];
//
//      // Input contribution
//      for (uint32_t i = 0; i < config->input_size; i++) 
//      {
//        sum += curr_layer->input_weights[i * config->hidden_size + h] * input_t[i];
//      }
//
//      // Previous hidden contribution
//      if (h_prev) 
//      {
//        for (uint32_t prev_h = 0; prev_h < config->hidden_size; prev_h++) 
//        {
//          sum += curr_layer->hidden_weights[prev_h * config->hidden_size + h] * h_prev[prev_h];
//        }
//      }
//      raw_hidden[h] = sum;
//    }
//
//    // Apply activation gradient
//    for (uint32_t h = 0; h < config->hidden_size; h++)
//    {
//      double activation_grad = FunctionGradient(curr_layer, raw_hidden[h]);
//      h_grad[h] *= activation_grad;
//    }
//
//    // Compute weight gradients
//    for (uint32_t h = 0; h < config->hidden_size; h++)
//    {
//      // Bias gradient
//      bias_grad[h] += h_grad[h];
//
//      // Input weight gradient (Wxh)
//      for (uint32_t i = 0; i < config->input_size; i++)
//      {
//        wxh_grad[i * config->hidden_size + h] += h_grad[h] * input_t[i];
//      }
//
//      // Hidden weight gradient (Whh)
//      if (h_prev)
//      {
//        for (uint32_t prev_h = 0; prev_h < config->hidden_size; prev_h++) 
//        {
//          whh_grad[prev_h * config->hidden_size + h] += h_grad[h] * h_prev[prev_h];
//        }
//      }
//    }
//
//    // Propagate error to input
//    for (uint32_t i = 0; i < config->input_size; i++)
//    {
//      for (uint32_t h = 0; h < config->hidden_size; h++) 
//      {
//        prev->errors[t * config->input_size + i] += h_grad[h] * curr_layer->input_weights[i * config->hidden_size + h];
//      }
//    }
//
//    // Propagate error to previous hidden state
//    if (t > 0) {
//      for (uint32_t prev_h = 0; prev_h < config->hidden_size; prev_h++)
//      {
//        next_h_grad[prev_h] = 0.0;
//        for (uint32_t h = 0; h < config->hidden_size; h++)
//        {
//          next_h_grad[prev_h] += h_grad[h] * curr_layer->hidden_weights[prev_h * config->hidden_size + h];
//        }
//      }
//    }
//  }
//
//  // Update weights using accumulated gradients
//  for (uint32_t i = 0; i < config->input_size * config->hidden_size; i++)
//  {
//    curr_layer->input_weights[i] -= learningRate_ * wxh_grad[i];
//  }
//
//  for (uint32_t i = 0; i < config->hidden_size * config->hidden_size; i++)
//  {
//    curr_layer->hidden_weights[i] -= learningRate_ * whh_grad[i];
//  }
//
//
//  for (uint32_t i = 0; i < config->hidden_size; i++) {
//    curr_layer->bias_weights[i] -= learningRate_ * bias_grad[i];
//  }
//}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

Layer* Model::GetLayer(size_t index)
{
  Layer* layer = nullptr;
  if (index < layers_.size())
    layer = layers_.at(index);

  return layer;
}

//---------------------------------------------------------------------------------------------------------------

void Model::ForwardPathFull(Layer* curr_layer)
{
  Layer* prev = PreviousLayer(curr_layer);

  // Initialize outputs to zero
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

      // Apply activation function and dropout during training
      if (training_)
      {
        curr_layer->outputs[j] = NonLinearFunction(curr_layer, val) * curr_layer->dropOut[j];
        if (curr_layer->layer_num != layers_.size() - 1 && curr_layer->type == Layer::fullLayer && (curr_layer->layer_num != 0))
        {
          GenerateDroputNodes(curr_layer);
        }
      }
      else
      {
        curr_layer->outputs[j] = NonLinearFunction(curr_layer, val);
      }
      curr_layer->gradients[j] = FunctionGradient(curr_layer, val);
    }

    // Apply residual connection if configured
    if (curr_layer->has_residual_connection)
    {
      // Get the source layer for the residual connection
      Layer* source_layer;
      if (curr_layer->residual_source_layer == -1)
      {
        source_layer = prev;
      }
      else
      {
        source_layer = GetLayer(curr_layer->residual_source_layer);
      }

      // Add the residual connection
      if (source_layer->n_outputs == curr_layer->n_outputs)
      {
        // Direct addition if dimensions match
        for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
        {
          curr_layer->outputs[i] += source_layer->outputs[i];
        }
      }
    }
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::BackwardPathFull(Layer* curr_layer)
{
  if (curr_layer->type != Layer::inputLayer)
  {
    Layer* prev = layers_.at(curr_layer->layer_num - 1);
    if (prev != NULL)
    {
      for (uint32_t i = 0; i < prev->n_outputs; i++)
      {
        int k = i;
        for (uint32_t j = 0; j < curr_layer->n_outputs; j++)
        {
          prev->errors[i] += curr_layer->weights[k] * curr_layer->errors[j] /* curr_layer->gradients[j]*/;
          k += prev->n_outputs;
        }
      }

      // Handle residual connection in backpropagation
      if (curr_layer->has_residual_connection)
      {
        // Get the source layer for the residual connection
        Layer* source_layer;
        if (curr_layer->residual_source_layer == -1)
        {
          source_layer = prev;
        }
        else
        {
          source_layer = GetLayer(curr_layer->residual_source_layer);
        }

        // Direct error propagation through skip connection if dimensions match
        if (source_layer->n_outputs == curr_layer->n_outputs)
        {
          for (uint32_t i = 0; i < source_layer->n_outputs; i++)
          {
            // Direct gradient path (the +1 term): errors flow directly through skip connection
            source_layer->errors[i] += curr_layer->errors[i];
          }
        }
      }

      // Calculate weight updates (gradient descent)
      int k = 0;
      for (uint32_t j = 0; j < curr_layer->n_outputs; j++)
      {
        for (uint32_t i = 0; i < prev->n_outputs; i++)
        {
          // Accumulate gradients (add to existing u_weights)
          //curr_layer->u_weights[k++] += curr_layer->errors[j] * curr_layer->gradients[j] * prev->outputs[i];
          double current_gradient = curr_layer->errors[j] * curr_layer->gradients[j] * prev->outputs[i];
          curr_layer->velocities[k] = momentum_ * curr_layer->velocities[k] + learningRate_ * current_gradient;
          curr_layer->u_weights[k] += curr_layer->velocities[k];
          k++;
        }
      }

      currentBatchCount_++;
      if (currentBatchCount_ >= batchSize_)
      {
        for (uint32_t i = 0; i < curr_layer->n_weights; i++)
        {
          double avg_gradient = curr_layer->u_weights[i] / batchSize_;
          curr_layer->weights[i] -= learningRate_ * avg_gradient;
          curr_layer->u_weights[i] = 0.0;
        }
      }
    }
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::ForwardPathAttention(Layer* curr_layer)
{
  Layer* prev = PreviousLayer(curr_layer);
  Layer::AttentionConfig* config = curr_layer->attention_config;

  uint32_t seq_len = config->seq_length;
  uint32_t d_model = config->d_model;
  uint32_t d_k = config->d_k;
  uint32_t d_v = config->d_v;
  uint32_t num_heads = config->num_heads;

  // Initialize outputs to zero
  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    curr_layer->outputs[i] = 0.0;
  }

  // For each attention head
  for (uint32_t head = 0; head < num_heads; head++) 
  {
    uint32_t head_offset_qk = head * seq_len * d_k;
    uint32_t head_offset_v = head * seq_len * d_v;
    uint32_t head_offset_scores = head * seq_len * seq_len;

    // Step 1: Compute Q, K, V matrices
    // Q = X * W_Q, K = X * W_K, V = X * W_V
    for (uint32_t i = 0; i < seq_len; i++) 
    {
      // Compute Q[i] = X[i] * W_Q
      for (uint32_t j = 0; j < d_k; j++) 
      {
        curr_layer->temp_q[head_offset_qk + i * d_k + j] = 0.0;
        for (uint32_t k = 0; k < d_model; k++) 
        {
          uint32_t weight_idx = head * d_model * d_k + j * d_model + k;
          curr_layer->temp_q[head_offset_qk + i * d_k + j] +=  prev->outputs[i * d_model + k] * curr_layer->query_weights[weight_idx];
        }
      }

      // Compute K[i] = X[i] * W_K
      for (uint32_t j = 0; j < d_k; j++) 
      {
        curr_layer->temp_k[head_offset_qk + i * d_k + j] = 0.0;
        for (uint32_t k = 0; k < d_model; k++) 
        {
          uint32_t weight_idx = head * d_model * d_k + j * d_model + k;
          curr_layer->temp_k[head_offset_qk + i * d_k + j] += prev->outputs[i * d_model + k] * curr_layer->key_weights[weight_idx];
        }
      }

      // Compute V[i] = X[i] * W_V
      for (uint32_t j = 0; j < d_v; j++) 
      {
        curr_layer->temp_v[head_offset_v + i * d_v + j] = 0.0;
        for (uint32_t k = 0; k < d_model; k++) 
        {
          uint32_t weight_idx = head * d_model * d_v + j * d_model + k;
          curr_layer->temp_v[head_offset_v + i * d_v + j] +=  prev->outputs[i * d_model + k] * curr_layer->value_weights[weight_idx];
        }
      }
    }

    // Step 2: Compute attention scores Q * K^T / sqrt(d_k)
    double scale_factor = 1.0 / sqrt((double)d_k);
    for (uint32_t i = 0; i < seq_len; i++) 
    {
      for (uint32_t j = 0; j < seq_len; j++)
      {
        double score = 0.0;
        for (uint32_t k = 0; k < d_k; k++) 
        {
          score += curr_layer->temp_q[head_offset_qk + i * d_k + k] *
                   curr_layer->temp_k[head_offset_qk + j * d_k + k];
        }
        curr_layer->attention_scores[head_offset_scores + i * seq_len + j] = score * scale_factor;
      }
    }

    // Step 3: Apply softmax to get attention weights
    for (uint32_t i = 0; i < seq_len; i++)
    {
      // Find max for numerical stability
      double max_score = curr_layer->attention_scores[head_offset_scores + i * seq_len];
      for (uint32_t j = 1; j < seq_len; j++) 
      {
        double score = curr_layer->attention_scores[head_offset_scores + i * seq_len + j];
        if (score > max_score) max_score = score;
      }

      // Compute softmax
      double sum = 0.0;
      for (uint32_t j = 0; j < seq_len; j++)
      {
        double exp_score = exp(curr_layer->attention_scores[head_offset_scores + i * seq_len + j] - max_score);
        curr_layer->attention_weights[head_offset_scores + i * seq_len + j] = exp_score;
        sum += exp_score;
      }

      // Normalize
      for (uint32_t j = 0; j < seq_len; j++)
      {
        curr_layer->attention_weights[head_offset_scores + i * seq_len + j] /= sum;
      }
    }

    // Step 4: Compute weighted sum of values
    double* head_output = new double[seq_len * d_v];
    for (uint32_t i = 0; i < seq_len * d_v; i++) head_output[i] = 0.0;

    for (uint32_t i = 0; i < seq_len; i++)
    {
      for (uint32_t j = 0; j < d_v; j++) 
      {
        for (uint32_t k = 0; k < seq_len; k++)
        {
          head_output[i * d_v + j] +=
            curr_layer->attention_weights[head_offset_scores + i * seq_len + k] *
            curr_layer->temp_v[head_offset_v + k * d_v + j];
        }
      }
    }

    // Step 5: Apply output projection W_O
    for (uint32_t i = 0; i < seq_len; i++)
    {
      for (uint32_t j = 0; j < d_model; j++)
      {
        for (uint32_t k = 0; k < d_v; k++) 
        {
          uint32_t weight_idx = head * d_v * d_model + j * d_v + k;
          curr_layer->outputs[i * d_model + j] +=
            head_output[i * d_v + k] * curr_layer->output_weights[weight_idx];
        }
      }
    }

    delete[] head_output;
  }

  // For multi-head, we've already summed the outputs
  // For single head, outputs are already computed

  // Store gradients (identity for attention - we'll compute actual gradients in backprop)
  for (uint32_t i = 0; i < curr_layer->n_outputs; i++)
  {
    curr_layer->gradients[i] = 1.0;  // Identity gradient for attention mechanism
  }
}

//---------------------------------------------------------------------------------------------------------------

// Attention backward pass
void Model::BackwardPathAttention(Layer* curr_layer)
{
  if (curr_layer->type != Layer::inputLayer) 
  {
    Layer* prev = layers_.at(curr_layer->layer_num - 1);
    Layer::AttentionConfig* config = curr_layer->attention_config;

    uint32_t seq_len = config->seq_length;
    uint32_t d_model = config->d_model;
    uint32_t d_k = config->d_k;
    uint32_t d_v = config->d_v;
    uint32_t num_heads = config->num_heads;

    // Initialize previous layer errors
    for (uint32_t i = 0; i < prev->n_outputs; i++) 
    {
      prev->errors[i] = 0.0;
    }

    // Initialize weight gradients
    for (uint32_t i = 0; i < d_model * d_k * num_heads; i++) 
    {
      curr_layer->query_gradients[i] = 0.0;
      curr_layer->key_gradients[i] = 0.0;
    }
    for (uint32_t i = 0; i < d_model * d_v * num_heads; i++)
    {
      curr_layer->value_gradients[i] = 0.0;
    }
    for (uint32_t i = 0; i < d_v * d_model * num_heads; i++) 
    {
      curr_layer->output_gradients[i] = 0.0;
    }

    double scale_factor = 1.0 / sqrt((double)d_k);

    // For each attention head
    for (uint32_t head = 0; head < num_heads; head++) 
    {
      uint32_t head_offset_qk = head * seq_len * d_k;
      uint32_t head_offset_v = head * seq_len * d_v;
      uint32_t head_offset_scores = head * seq_len * seq_len;

      // Temporary arrays for gradients
      double* grad_v = new double[seq_len * d_v];
      double* grad_scores = new double[seq_len * seq_len];
      double* grad_q = new double[seq_len * d_k];
      double* grad_k = new double[seq_len * d_k];

      // Initialize
      for (uint32_t i = 0; i < seq_len * d_v; i++) grad_v[i] = 0.0;
      for (uint32_t i = 0; i < seq_len * seq_len; i++) grad_scores[i] = 0.0;
      for (uint32_t i = 0; i < seq_len * d_k; i++) 
      {
        grad_q[i] = 0.0;
        grad_k[i] = 0.0;
      }

      // Backprop through output projection
      for (uint32_t i = 0; i < seq_len; i++) 
      {
        for (uint32_t j = 0; j < d_model; j++) 
        {
          for (uint32_t k = 0; k < d_v; k++)
          {
            uint32_t weight_idx = head * d_v * d_model + j * d_v + k;

            // Gradient w.r.t. output weights
            curr_layer->output_gradients[weight_idx] +=
              curr_layer->errors[i * d_model + j] * curr_layer->temp_v[head_offset_v + i * d_v + k];

            // Gradient w.r.t. values
            grad_v[i * d_v + k] +=
              curr_layer->errors[i * d_model + j] * curr_layer->output_weights[weight_idx];
          }
        }
      }

      // Backprop through attention weights to values
      for (uint32_t i = 0; i < seq_len; i++) 
      {
        for (uint32_t j = 0; j < d_v; j++) 
        {
          for (uint32_t k = 0; k < seq_len; k++)
          {
            // Gradient w.r.t. attention scores
            grad_scores[i * seq_len + k] +=
              grad_v[i * d_v + j] * curr_layer->temp_v[head_offset_v + k * d_v + j];
          }
        }
      }

      // Backprop through softmax
      for (uint32_t i = 0; i < seq_len; i++)
      {
        for (uint32_t j = 0; j < seq_len; j++)
        {
          double softmax_grad = 0.0;
          for (uint32_t k = 0; k < seq_len; k++)
          {
            double delta_jk = (j == k) ? 1.0 : 0.0;
            softmax_grad += grad_scores[i * seq_len + k] *
              curr_layer->attention_weights[head_offset_scores + i * seq_len + j] *
              (delta_jk - curr_layer->attention_weights[head_offset_scores + i * seq_len + k]);
          }
          grad_scores[i * seq_len + j] = softmax_grad * scale_factor;
        }
      }

      // Backprop through Q*K^T to Q and K
      for (uint32_t i = 0; i < seq_len; i++) 
      {
        for (uint32_t j = 0; j < seq_len; j++)
        {
          for (uint32_t k = 0; k < d_k; k++) 
          {
            // Gradient w.r.t. Q
            grad_q[i * d_k + k] +=
              grad_scores[i * seq_len + j] * curr_layer->temp_k[head_offset_qk + j * d_k + k];

            // Gradient w.r.t. K
            grad_k[j * d_k + k] +=
              grad_scores[i * seq_len + j] * curr_layer->temp_q[head_offset_qk + i * d_k + k];
          }
        }
      }

      // Backprop through linear projections to input and weights
      for (uint32_t i = 0; i < seq_len; i++) 
      {
        for (uint32_t j = 0; j < d_model; j++)
        {
          // Query gradients
          for (uint32_t k = 0; k < d_k; k++) {
            uint32_t weight_idx = head * d_model * d_k + k * d_model + j;
            curr_layer->query_gradients[weight_idx] +=
              grad_q[i * d_k + k] * prev->outputs[i * d_model + j];
            prev->errors[i * d_model + j] +=
              grad_q[i * d_k + k] * curr_layer->query_weights[weight_idx];
          }

          // Key gradients
          for (uint32_t k = 0; k < d_k; k++) 
          {
            uint32_t weight_idx = head * d_model * d_k + k * d_model + j;
            curr_layer->key_gradients[weight_idx] +=
              grad_k[i * d_k + k] * prev->outputs[i * d_model + j];
            prev->errors[i * d_model + j] +=
              grad_k[i * d_k + k] * curr_layer->key_weights[weight_idx];
          }

          // Value gradients
          for (uint32_t k = 0; k < d_v; k++) 
          {
            uint32_t weight_idx = head * d_model * d_v + k * d_model + j;
            curr_layer->value_gradients[weight_idx] +=
              grad_v[i * d_v + k] * prev->outputs[i * d_model + j];
            prev->errors[i * d_model + j] +=
              grad_v[i * d_v + k] * curr_layer->value_weights[weight_idx];
          }
        }
      }

      delete[] grad_v;
      delete[] grad_scores;
      delete[] grad_q;
      delete[] grad_k;
    }

    // Update weights using gradients
    for (uint32_t i = 0; i < d_model * d_k * num_heads; i++) 
    {
      curr_layer->query_weights[i] -= learningRate_ * curr_layer->query_gradients[i];
      curr_layer->key_weights[i] -= learningRate_ * curr_layer->key_gradients[i];
    }
    for (uint32_t i = 0; i < d_model * d_v * num_heads; i++) 
    {
      curr_layer->value_weights[i] -= learningRate_ * curr_layer->value_gradients[i];
    }
    for (uint32_t i = 0; i < d_v * d_model * num_heads; i++) 
    {
      curr_layer->output_weights[i] -= learningRate_ * curr_layer->output_gradients[i];
    }
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::ForwardPathSoftmax(Layer* curr_layer)
{
  Layer* prev_layer = PreviousLayer(curr_layer);
  double max_val = prev_layer->outputs[0];
  for (uint32_t i = 1; i < curr_layer->n_outputs; ++i)
  {
    if (prev_layer->outputs[i] > max_val)
    {
      max_val = prev_layer->outputs[i];
    }
  }
  double sum = 0.0;
  for (uint32_t i = 0; i < prev_layer->n_outputs; ++i)
  {
    curr_layer->outputs[i] = exp(prev_layer->outputs[i] - max_val);
    sum += curr_layer->outputs[i];
  }
  for (uint32_t i = 0; i < curr_layer->n_outputs; ++i)
  {
    curr_layer->outputs[i] /= sum;
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::BackwardPathSoftmax(Layer* curr_layer)
{
  Layer* prev_layer = PreviousLayer(curr_layer);
  if (prev_layer == nullptr) return;

  for (uint32_t i = 0; i < prev_layer->n_outputs; ++i)
  {
    prev_layer->errors[i] = curr_layer->errors[i];
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::ForwardPathMaxpool(Layer* curr_layer)
{
  Layer* prev_layer = PreviousLayer(curr_layer);
  uint32_t output_height = (prev_layer->height - curr_layer->kern_size) / curr_layer->stride + 1;
  uint32_t output_width = (prev_layer->width - curr_layer->kern_size) / curr_layer->stride + 1;

  // Store max indices for backward pass
  if (!curr_layer->max_indices)
  {
    curr_layer->max_indices = new double[output_height * output_width * prev_layer->depth];
  }

  int output_idx = 0;

  for (uint32_t z = 0; z < prev_layer->depth; z++)
  {
    for (uint32_t y = 0; y < output_height; y++)
    {
      for (uint32_t x = 0; x < output_width; x++)
      {
        double max_value = -DBL_MAX;  // Initialize to very small value
        int max_index = -1;

        // Find maximum in the kernel window
        for (uint32_t ky = 0; ky < curr_layer->kern_size; ky++)
        {
          int y_pos = y * curr_layer->stride + ky;
          for (uint32_t kx = 0; kx < curr_layer->kern_size; kx++)
          {
            int x_pos = x * curr_layer->stride + kx;
            int input_idx = (z * prev_layer->height * prev_layer->width) + (y_pos * prev_layer->width) + x_pos;

            if (prev_layer->outputs[input_idx] > max_value)
            {
              max_value = prev_layer->outputs[input_idx];
              max_index = input_idx;
            }
          }
        }

        curr_layer->outputs[output_idx] = max_value;
        curr_layer->max_indices[output_idx] = max_index;  // Store for backward pass
        output_idx++;
      }
    }
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::BackwardPathMaxpool(Layer* curr_layer)
{
  Layer* prev_layer = PreviousLayer(curr_layer);

  // Clear previous layer errors
  memset(prev_layer->errors, 0, prev_layer->n_outputs * sizeof(double));

  uint32_t output_height = (prev_layer->height - curr_layer->kern_size) / curr_layer->stride + 1;
  uint32_t output_width = (prev_layer->width - curr_layer->kern_size) / curr_layer->stride + 1;
  int total_outputs = output_height * output_width * prev_layer->depth;

  // Propagate error to the neuron that was selected during forward pass
  for (int i = 0; i < total_outputs; i++)
  {
    int max_input_idx = curr_layer->max_indices[i];
    prev_layer->errors[max_input_idx] += curr_layer->errors[i];
  }
}

//---------------------------------------------------------------------------------------------------------------
// 
//void Model::BackwardPathMaxpool(Layer* curr_layer)
//{
//  Layer* prev = PreviousLayer(curr_layer);
//  int k = 0;
//  memset(prev->errors, 0, prev->n_outputs * sizeof(double));
//
//  uint32_t output_height = (prev->height - curr_layer->kern_size) / curr_layer->stride + 1;
//  uint32_t output_width = (prev->width - curr_layer->kern_size) / curr_layer->stride + 1;
//
//  //    PrevLayer                      CurrLayer(maxpool)
//  // 
//  //    n1(0.8)-----------
//  //                      |
//  //    n2 (0.0)----------
//  //                       \___________n(max)(e.g)error = 0.8)
//  //    n3(0.0)------------/
//  //                      |
//  //    n4(0.0)------------
//  //
//  // the max pool back propagation error propagates its errors to the max node cell that resulted in its value, leaving the rest to 0s.
//  //
//  // This layer has no weights, so just we need to back propagate the errors.
//
//  for (uint32_t z = 0; z < prev->depth; z++)
//  {
//    for (uint32_t y = 0; y < output_height/*prev->height - curr_layer->kern_size*/; y++)
//    {
//      for (uint32_t x = 0; x < output_width/*prev->width - curr_layer->kern_size*/; x++)
//      {
//        int max_index = (z * prev->height * prev->width) + (y * curr_layer->stride * prev->width) + (x * curr_layer->stride);
//        double max_value = prev->outputs[max_index];// previous_output[z][y * curr_layer->stride][x * curr_layer->stride];// assume first node to be the max.
//        int max_y = y * curr_layer->stride, max_x = x * curr_layer->stride;
//
//        for (uint32_t m = 0; m < curr_layer->kern_size; m++)
//        {
//          int y0 = y * curr_layer->stride + m;
//          for (uint32_t n = 0; n < curr_layer->kern_size; n++)
//          {
//            int x0 = x * curr_layer->stride + n;
//            int prev_out_index = (z * prev->height * prev->width) + (y0 * prev->width) + x0;
//            if (prev->outputs[prev_out_index] >= max_value)
//            {
//              max_value = prev->outputs[prev_out_index];// previous_output[z][y0][x0];
//              max_y = y0;
//              max_x = x0;
//            }
//            if (m == curr_layer->kern_size - 1 && n == (curr_layer->kern_size - 1))
//            {
//              int prev_index = (z * prev->height * prev->width) + (max_y * prev->width) + max_x;
//              prev->errors[prev_index] = curr_layer->errors[k++];
//            }
//          }
//        }
//      }
//    }
//  }
//}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

void Model::ForwardPath(Layer* layer)
{
  if (layer->type == Layer::fullLayer)
  {
    ForwardPathFull(layer);
  }
  else if (layer->type == Layer::attentionLayer) 
  {
    ForwardPathAttention(layer);
  }
  else if (layer->type == Layer::convLayer)
  {
    ForwardPathConv(layer);
  }
  else if (layer->type == Layer::maxpoolLayer)
  {
    ForwardPathMaxpool(layer);
  }
  else if (layer->type == Layer::normalizationLayer)
  {
    ForwardPathLayerNorm(layer);
  }
  else if (layer->type == Layer::softmax)
  {
    ForwardPathSoftmax(layer);
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::TrainData(std::vector<double> values)
{
  Layer* curr = GetLayer(layers_.size() - 1);
  Layer* prev = PreviousLayer(curr);

  for (auto& layer : layers_)
  {
    for (uint32_t i = 0; i < layer->n_outputs; i++)
    {
      layer->errors[i] = 0;
    }
  }

  for (uint32_t i = 0; i < curr->n_outputs; i++)
  {
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
    else if (curr->type == Layer::residualFullLayer)
    {
      BackwardPathFull(curr);
    }
    else if (curr->type == Layer::attentionLayer) 
    {
      BackwardPathAttention(curr);
    }
    else if (curr->type == Layer::softmax)
    {
      BackwardPathSoftmax(curr);
    }
    else
    {
      BackwardPathFull(curr);
    }

    curr = PreviousLayer(curr);
  }
  if (currentBatchCount_ >= batchSize_)
  {
    currentBatchCount_ = 0;
  }
}

//---------------------------------------------------------------------------------------------------------------

void Model::FlushBatch()
{
  if (currentBatchCount_ > 0) 
  {
    // Force update with partial batch
    for (auto& layer : layers_) 
    {
      if (layer->u_weights != nullptr) 
      {
        for (uint32_t i = 0; i < layer->n_weights; i++) 
        {
          double avg_gradient = layer->u_weights[i] / currentBatchCount_;
          layer->weights[i] -= learningRate_ * avg_gradient;
          layer->u_weights[i] = 0.0;
        }
      }
    }
    currentBatchCount_ = 0;
  }
}

//---------------------------------------------------------------------------------------------------------------

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
    ForwardPath(curr);
    curr = NextLayer(curr);
  }
}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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
      std::string type = (layers_.at(i)->type == Layer::fullLayer) ? "FullLayer" :
        (layers_.at(i)->type == Layer::inputLayer) ? "InputLayer" :
        (layers_.at(i)->type == Layer::maxpoolLayer) ? "MaxpoolLayer" :
        (layers_.at(i)->type == Layer::normalizationLayer) ? "NormalizationLayer" : "ConvLayer";

      std::string func = (layers_.at(i)->funct == Layer::sigmoid) ? "sigmoid" :
        (layers_.at(i)->funct == Layer::tanh) ? "tanh" :
        (layers_.at(i)->funct == Layer::leakyRelu) ? "leakyrelu" : "relu";

      if (type == "ConvLayer" || type == "InputLayer")
      {
        fprintf(file, "Layer%d_Nodes = %d, type = %s, function = %s, height = %d, width = %d, depth = %d, kern_size = %d, stride = %d, padding = %d\n",
          i, layers_.at(i)->n_outputs, type.c_str(), func.c_str(), layers_.at(i)->height, layers_.at(i)->width, layers_.at(i)->depth, layers_.at(i)->kern_size, layers_.at(i)->stride, layers_.at(i)->padding);
      }
      else if (type == "FullLayer")
      {
        // Check if this is a residual layer
        if (layers_.at(i)->has_residual_connection)
        {
          fprintf(file, "Layer%d_Nodes = %d, type = %s, function = %s, isResidual = true, residual_from = %d\n", i, layers_.at(i)->n_outputs, type.c_str(), func.c_str(), layers_.at(i)->residual_source_layer);
        }
        else
        {
          fprintf(file, "Layer%d_Nodes = %d, type = %s, function = %s\n", i, layers_.at(i)->n_outputs, type.c_str(), func.c_str());
        }
      }
      else if (type == "MaxpoolLayer")
      {
        fprintf(file, "Layer%d_Nodes = %d, type = %s, kern_size = %d, stride = %d\n", i, layers_.at(i)->n_outputs, type.c_str(), layers_.at(i)->kern_size, layers_.at(i)->stride);
      }
      else if (type == "NormalizationLayer")
      {
        fprintf(file, "Layer%d_Nodes = %d, type = %s\n", i, layers_.at(i)->n_outputs, type.c_str());
      }
    }
    fclose(file);
  }
  else
  {
    printf("Error opening the file for writing\n");
  }

  SaveWeights(modelName_);
  return true;
}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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
    uint32_t numLayers, layerNodes[40], layerHeight[40], layerWidth[40], layerDepth[40], layerKernSize[40], layerStride[40], layerPadding[40], residualFromLayer[40];
    bool isResidual[40];
    char stype[40];
    char sfunction[40];
    std::vector<std::string> types;
    std::vector<std::string> functions;
    int k = 0;

    // Initialize arrays
    memset(layerNodes, 0, sizeof(layerNodes));
    memset(layerHeight, 0, sizeof(layerHeight));
    memset(layerWidth, 0, sizeof(layerWidth));
    memset(layerDepth, 0, sizeof(layerDepth));
    memset(layerKernSize, 0, sizeof(layerKernSize));
    memset(layerStride, 0, sizeof(layerStride));
    memset(layerPadding, 0, sizeof(layerPadding));
    memset(residualFromLayer, 0, sizeof(residualFromLayer));
    memset(isResidual, false, sizeof(isResidual));

    while (fgets(line, sizeof(line), file))
    {
      if (sscanf(line, "NumberOfLayers = %d\n", &numLayers) == 1)
      {
        //printf("Number of Layers: %d\n", numLayers);
      }
      // ConvLayer or InputLayer with all parameters
      else if (sscanf(line, "Layer%d_Nodes = %d, type = %19[^,], function = %19[^,], height = %d, width = %d, depth = %d, kern_size = %d, stride = %d, padding = %d\n",
        &k, &layerNodes[k], stype, sfunction, &layerHeight[k], &layerWidth[k], &layerDepth[k], &layerKernSize[k], &layerStride[k], &layerPadding[k]) == 10)
      {
        types.push_back(stype);
        functions.push_back(sfunction);
        k++;
      }
      // ResidualFullLayer with skip connection
      else if (sscanf(line, "Layer%d_Nodes = %d, type = %19[^,], function = %19[^,], isResidual = true, residual_from = %d\n",
        &k, &layerNodes[k], stype, sfunction, &residualFromLayer[k]) == 5)
      {
        types.push_back(stype);
        functions.push_back(sfunction);
        isResidual[k] = true;
        k++;
      }
      // FullLayer with type and function
      else if (sscanf(line, "Layer%d_Nodes = %d, type = %19[^,], function = %19[^\n]",
        &k, &layerNodes[k], stype, sfunction) == 4)
      {
        types.push_back(stype);
        functions.push_back(sfunction);
        k++;
      }
      // MaxpoolLayer with kernel size and stride
      else if (sscanf(line, "Layer%d_Nodes = %d, type = %19[^,], kern_size = %d, stride = %d\n",
        &k, &layerNodes[k], stype, &layerKernSize[k], &layerStride[k]) == 5)
      {
        types.push_back(stype);
        functions.push_back("none"); // MaxpoolLayer doesn't have activation function
        k++;
      }
      // NormalizationLayer (simple case)
      else if (sscanf(line, "Layer%d_Nodes = %d, type = %19[^\n]",
        &k, &layerNodes[k], stype) == 3)
      {
        types.push_back(stype);
        functions.push_back("none"); // NormalizationLayer doesn't have activation function
        k++;
      }
    }

    for (uint32_t i = 0; i < numLayers; i++)
    {
      Layer::LayerType type = (types[i] == "ConvLayer") ? Layer::LayerType::convLayer :
        (types[i] == "InputLayer") ? Layer::LayerType::inputLayer :
        (types[i] == "MaxpoolLayer") ? Layer::LayerType::maxpoolLayer :
        (types[i] == "NormalizationLayer") ? Layer::LayerType::normalizationLayer :
        Layer::LayerType::fullLayer;

      Layer::ActivationFunc func = (functions[i] == "sigmoid") ? Layer::ActivationFunc::sigmoid :
        (functions[i] == "relu") ? Layer::ActivationFunc::relu :
        (functions[i] == "leakyrelu") ? Layer::ActivationFunc::leakyRelu :
        (functions[i] == "tanh") ? Layer::ActivationFunc::tanh :
        Layer::ActivationFunc::relu; // default

      if (type == Layer::fullLayer)
      {
        CreateFullLayer(layerNodes[i], func);

        // If this is a residual layer, set the residual properties after creation
        if (isResidual[i])
        {
          layers_.back()->has_residual_connection = true;
          layers_.back()->residual_source_layer = residualFromLayer[i];
        }
      }
      else if (type == Layer::inputLayer)
      {
        CreateInputLayer(layerHeight[i], layerWidth[i], layerDepth[i]);
      }
      else if (type == Layer::convLayer)
      {
        CreateConvLayer(layerDepth[i], layerKernSize[i], layerStride[i], layerPadding[i], func);
      }
      else if (type == Layer::maxpoolLayer)
      {
        CreateMaxpoolLayer(layerKernSize[i], layerStride[i]); // Fixed: was layerKernSize[i], layerKernSize[i]
      }
      else if (type == Layer::normalizationLayer)
      {
        CreateNormalizationLayer();
      }
    }
    fclose(file); // Close the file
  }
  else
  {
    printf("Error opening the file %s for reading\n", modelPath.c_str());
  }

  if (modelExist)
    modelExist &= LoadWeights(Model);

  return modelExist;
}

//---------------------------------------------------------------------------------------------------------------

bool Model::LoadWeights(std::string Model)
{
  bool loaded = true;
  if (Model.empty())
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

    fp = fopen(layerWeights.c_str(), "r");
    if (fp == NULL)
    {
      printf("Error opening file\n");
      return false;
    }
    char line[256];
    int k = 0;
    double threshold = 0.3;// GetMinimumThreshold(layerWeights, 0.5);
    uint32_t numOfActiveWeights = 0;
    for (int i = 0; i < current->n_weights; i++)
    {
      float weight;
      fscanf(fp, "%g", &weight);
      if (pruning_ == true)
      {
        current->weights[k] = (abs(weight) > pruningThreshold_) ? weight : 0.0; //pruning
        numOfActiveWeights = (abs(weight) > pruningThreshold_) ? (numOfActiveWeights + 1) : numOfActiveWeights;
        k++;
      }
      else
      {
        current->weights[k] = weight;
        numOfActiveWeights++;
        k++;
      }
    }
    printf("Layer %d, num of weights = %d, number of active weights = %d\n", current->layer_num, current->n_weights, numOfActiveWeights);
    fclose(fp);
    current = NextLayer(current);
  }

  return loaded;
}

//---------------------------------------------------------------------------------------------------------------

void Model::ReadCsvFile(std::string fileName, std::vector<std::vector<double>>& inputs, std::vector<uint32_t>& outputs, bool scaled)
{
  std::ifstream file(fileName.c_str());
  std::string line;

  while (std::getline(file, line))
  {
    std::vector<double> values;
    std::stringstream ss(line);
    std::string value;

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

void Model::LearnData(std::vector<double> inputs, std::vector<double> outputs)
{
  training_ = true;
  InputData(inputs);
  TrainData(outputs);

  double errors = GetTotalError();
  m_totalError += errors;
  training_ = false;
}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

double Model::GetAverageError(uint32_t epochs)
{
  double currentError = m_totalError / epochs;
  m_totalError = 0.0;
  return currentError;
}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

void  Model::AddClass(uint32_t index, std::string label)
{
  Classify newclass;
  newclass.index = index;
  newclass.label = label;
  classes_.push_back(newclass);
}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

float Model::GetCurrentConfidence()
{
  return confidence_;
}

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------

bool Model::SaveModelBinary(const std::string& filename)
{
  size_t lastSlash = filename.find_last_of("/\\");
  size_t lastDot = filename.find_last_of('.');
  if (lastSlash == std::string::npos) lastSlash = 0; else lastSlash++;
  if (lastDot == std::string::npos) lastDot = filename.length();
  modelName_ = filename.substr(lastSlash, lastDot - lastSlash);

  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    printf("Error: Cannot open file %s for writing\n", filename.c_str());
    return false;
  }

  {
    // Write header
    ModelHeader header = {};
    header.magic_number = 0x4D4F444C;  // "MODL"
    header.version = 1;
    header.num_layers = static_cast<uint32_t>(layers_.size());

    // Calculate total weights
    header.total_weights = 0;
    for (const auto& layer : layers_)
    {
      header.total_weights += layer->n_weights;
    }

    strncpy(header.model_name, modelName_.c_str(), sizeof(header.model_name) - 1);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write layer information
    for (const auto& layer : layers_)
    {
      LayerInfo info = {};
      info.layer_type = static_cast<uint32_t>(layer->type);
      info.activation_func = static_cast<uint32_t>(layer->funct);
      info.n_outputs = layer->n_outputs;
      info.n_weights = layer->n_weights;
      info.height = layer->height;
      info.width = layer->width;
      info.depth = layer->depth;
      info.kern_size = layer->kern_size;
      info.stride = layer->stride;
      info.padding = layer->padding;
      info.has_residual_connection = layer->has_residual_connection;
      info.residual_source_layer = layer->residual_source_layer;

      file.write(reinterpret_cast<const char*>(&info), sizeof(info));
    }

    // Write weights for all layers
    for (const auto& layer : layers_)
    {
      if (layer->n_weights > 0)
      {
        file.write(reinterpret_cast<const char*>(layer->weights), layer->n_weights * sizeof(double));
      }
    }

    file.close();
    return true;
  }
}

//---------------------------------------------------------------------------------------------------------------

bool Model::LoadModelBinary(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    printf("Error: Cannot open file %s for reading\n", filename.c_str());
    return false;
  }

  // Read and validate header
  ModelHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(header));

  if (file.gcount() != sizeof(header))
  {
    printf("Error: Invalid file format - header incomplete\n");
    return false;
  }

  if (header.magic_number != 0x4D4F444C)
  {
    printf("Error: Invalid file format - wrong magic number\n");
    return false;
  }

  if (header.version != 1)
  {
    printf("Error: Unsupported file version %d\n", header.version);
    return false;
  }

  printf("Loading model: %s\n", header.model_name);
  printf("Layers: %d, Total weights: %d\n", header.num_layers, header.total_weights);

  // Clear existing model
  layers_.clear();
  layerNum_ = 0;
  modelName_ = header.model_name;

  // Read layer information and create layers
  std::vector<LayerInfo> layer_infos(header.num_layers);
  for (uint32_t i = 0; i < header.num_layers; i++)
  {
    file.read(reinterpret_cast<char*>(&layer_infos[i]), sizeof(LayerInfo));

    if (file.gcount() != sizeof(LayerInfo))
    {
      printf("Error: Invalid file format - layer info incomplete\n");
      return false;
    }
  }

  // Create layers based on loaded information
  for (uint32_t i = 0; i < header.num_layers; i++)
  {
    const LayerInfo& info = layer_infos[i];
    Layer::LayerType type = static_cast<Layer::LayerType>(info.layer_type);
    Layer::ActivationFunc func = static_cast<Layer::ActivationFunc>(info.activation_func);

    switch (type)
    {
    case Layer::inputLayer:
      CreateInputLayer(info.height, info.width, info.depth);
      break;

    case Layer::fullLayer:
      CreateFullLayer(info.n_outputs, func);
      if (info.has_residual_connection)
      {
        layers_.back()->has_residual_connection = true;
        layers_.back()->residual_source_layer = info.residual_source_layer;
      }
      break;

    case Layer::convLayer:
      CreateConvLayer(info.depth, info.kern_size, info.stride, info.padding, func);
      break;

    case Layer::maxpoolLayer:
      CreateMaxpoolLayer(info.kern_size, info.stride);
      break;

    case Layer::normalizationLayer:
      CreateNormalizationLayer();
      break;

    default:
      printf("Error: Unknown layer type %d\n", info.layer_type);
      return false;
    }
  }

  // Read weights for all layers
  for (size_t i = 0; i < layers_.size(); i++)
  {
    Layer* layer = layers_[i];
    const LayerInfo& info = layer_infos[i];

    if (info.n_weights > 0)
    {
      if (layer->n_weights != info.n_weights)
      {
        printf("Error: Weight count mismatch in layer %d (expected %d, got %d)\n", layer->layer_num, info.n_weights, layer->n_weights);
        return false;
      }

      file.read(reinterpret_cast<char*>(layer->weights), layer->n_weights * sizeof(double));

      if (file.gcount() != static_cast<std::streamsize>(layer->n_weights * sizeof(double)))
      {
        printf("Error: Incomplete weight data for layer %d\n", layer->layer_num);
        return false;
      }

      // Apply pruning if enabled
      if (pruning_)
      {
        uint32_t active_weights = 0;
        for (uint32_t j = 0; j < layer->n_weights; j++)
        {
          if (abs(layer->weights[j]) <= pruningThreshold_)
          {
            layer->weights[j] = 0.0f;
          }
          else
          {
            active_weights++;
          }
        }
        printf("Layer %d: %d/%d weights active after pruning\n", layer->layer_num, active_weights, layer->n_weights);
      }
      else
      {
        printf("Layer %d: %d weights loaded\n", layer->layer_num, layer->n_weights);
      }
    }
  }

  file.close();
  printf("Model loaded successfully from %s\n", filename.c_str());
  return true;
}

void Model::PrintModelSummary() const
{
  if (layers_.empty())
  {
    printf("Model is empty - no layers defined\n");
    return;
  }

  printf("\n");
  printf("================================================================================\n");
  printf("                              MODEL SUMMARY                                     \n");
  printf("================================================================================\n");
  printf("Model Name: %s\n", modelName_.empty() ? "Unnamed" : modelName_.c_str());
  printf("Total Layers: %zu\n", layers_.size());

  // Calculate total parameters
  uint64_t total_weights = 0;
  uint64_t trainable_weights = 0;
  for (const auto& layer : layers_)
  {
    total_weights += layer->n_weights;
    if (layer->type != Layer::inputLayer && layer->type != Layer::maxpoolLayer &&
      layer->type != Layer::normalizationLayer)
    {
      trainable_weights += layer->n_weights;
    }
  }

  printf("Total Parameters: %llu\n", total_weights);
  printf("Trainable Parameters: %llu\n", trainable_weights);
  printf("\n");

  // Print layer details table
  printf("+-----+-----------------+-------------+----------------+-----------+-------------+\n");
  printf("| #   | Layer Type      | Output Size | Activation     | Weights   | Details     |\n");
  printf("+-----+-----------------+-------------+----------------+-----------+-------------+\n");

  for (size_t i = 0; i < layers_.size(); i++)
  {
    const Layer* layer = layers_[i];
    std::string layer_type = GetLayerTypeString(layer->type);
    std::string activation = GetActivationString(layer->funct);
    std::string output_size = GetOutputSizeString(layer);
    std::string details = GetLayerDetailsString(layer);

    printf("| %-3zu | %-15s | %-11s | %-14s | %-9u | %-11s |\n",
      i, layer_type.c_str(), output_size.c_str(),
      activation.c_str(), layer->n_weights, details.c_str());
  }

  printf("+-----+-----------------+-------------+----------------+-----------+-------------+\n");
  printf("\n");

  // Print residual connections if any
  bool has_residuals = false;
  for (const auto& layer : layers_)
  {
    if (layer->has_residual_connection)
    {
      if (!has_residuals)
      {
        printf("Residual Connections:\n");
        has_residuals = true;
      }
      printf("  Layer %d -> Layer %d (skip connection)\n",
        layer->residual_source_layer, layer->layer_num);
    }
  }

  if (has_residuals) printf("\n");

  // Print pruning info if enabled
  if (pruning_)
  {
    printf("Pruning: ENABLED (threshold: %.6f)\n", pruningThreshold_);
    uint64_t active_weights = 0;
    for (const auto& layer : layers_)
    {
      for (uint32_t j = 0; j < layer->n_weights; j++)
      {
        if (abs(layer->weights[j]) > pruningThreshold_)
        {
          active_weights++;
        }
      }
    }
    printf("Active Weights: %llu/%llu (%.2f%%)\n",
      active_weights, total_weights,
      total_weights > 0 ? (100.0 * active_weights / total_weights) : 0.0);
    printf("\n");
  }

  printf("================================================================================\n");
}


void Model::PrintDetailedModel() const
{
  if (layers_.empty())
  {
    printf("Model is empty - no layers defined\n");
    return;
  }

  printf("\n");
  printf("================================================================================\n");
  printf("                           DETAILED MODEL REPORT                               \n");
  printf("================================================================================\n");
  printf("Model Name: %s\n", modelName_.empty() ? "Unnamed" : modelName_.c_str());
  printf("Total Layers: %zu\n\n", layers_.size());

  for (size_t i = 0; i < layers_.size(); i++)
  {
    const Layer* layer = layers_[i];

    printf("Layer %zu: %s\n", i, GetLayerTypeString(layer->type).c_str());
    printf("  Layer Number: %d\n", layer->layer_num);
    printf("  Outputs: %u\n", layer->n_outputs);
    printf("  Weights: %u\n", layer->n_weights);

    if (layer->type == Layer::inputLayer || layer->type == Layer::convLayer)
    {
      printf("  Dimensions: %ux%ux%u (HxWxD)\n",
        layer->height, layer->width, layer->depth);
    }

    if (layer->type == Layer::convLayer || layer->type == Layer::maxpoolLayer)
    {
      printf("  Kernel Size: %u\n", layer->kern_size);
      printf("  Stride: %u\n", layer->stride);
      if (layer->type == Layer::convLayer)
      {
        printf("  Padding: %u\n", layer->padding);
      }
    }

    if (layer->type != Layer::inputLayer && layer->type != Layer::maxpoolLayer && layer->type != Layer::normalizationLayer)
    {
      printf("  Activation: %s\n", GetActivationString(layer->funct).c_str());
    }

    if (layer->has_residual_connection)
    {
      printf("  Residual Connection: FROM Layer %u\n", layer->residual_source_layer);
    }

    if (layer->n_weights > 0)
    {
      // Calculate weight statistics
      double min_weight = layer->weights[0];
      double max_weight = layer->weights[0];
      double sum_weight = 0.0;
      uint32_t zero_weights = 0;

      for (uint32_t j = 0; j < layer->n_weights; j++)
      {
        double w = layer->weights[j];
        if (w < min_weight) min_weight = w;
        if (w > max_weight) max_weight = w;
        sum_weight += w;
        if (abs(w) <= 1e-9) zero_weights++;
      }

      double avg_weight = sum_weight / layer->n_weights;
      printf("  Weight Stats: min=%.6f, max=%.6f, avg=%.6f\n", min_weight, max_weight, avg_weight);
      printf("  Zero Weights: %u/%u (%.2f%%)\n", zero_weights, layer->n_weights, 100.0 * zero_weights / layer->n_weights);
    }

    printf("\n");
  }

  printf("================================================================================\n");
}


std::string Model::GetLayerTypeString(Layer::LayerType type) const
{
  switch (type)
  {
  case Layer::inputLayer:        return "Input";
  case Layer::fullLayer:         return "Dense/Full";
  case Layer::convLayer:         return "Convolution";
  case Layer::maxpoolLayer:      return "MaxPool";
  case Layer::normalizationLayer: return "Normalization";
  default:                       return "Unknown";
  }
}

std::string Model::GetActivationString(Layer::ActivationFunc func) const
{
  switch (func)
  {
  case Layer::sigmoid:    return "Sigmoid";
  case Layer::tanh:       return "Tanh";
  case Layer::relu:       return "ReLU";
  case Layer::leakyRelu:  return "LeakyReLU";
  default:                return "None";
  }
}

std::string Model::GetOutputSizeString(const Layer* layer) const
{
  if (layer->type == Layer::inputLayer || layer->type == Layer::convLayer)
  {
    return std::to_string(layer->height) + "x" + std::to_string(layer->width) + "x" + std::to_string(layer->depth);
  }
  else
  {
    return std::to_string(layer->n_outputs);
  }
}

std::string Model::GetLayerDetailsString(const Layer* layer) const
{
  switch (layer->type)
  {
  case Layer::convLayer:
    return "k=" + std::to_string(layer->kern_size) + ",s=" + std::to_string(layer->stride);
  case Layer::maxpoolLayer:
    return "k=" + std::to_string(layer->kern_size) + ",s=" + std::to_string(layer->stride);
  case Layer::fullLayer:
    return layer->has_residual_connection ? "Residual" : "Standard";
  default:
    return "-";
  }
}
