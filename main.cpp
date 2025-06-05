#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "Model.hpp"

int main()
{
  std::cout << "XOR Gate Neural Network Test\n";
  std::cout << "============================\n\n";

  // XOR training data
  std::vector<std::vector<double>> xor_inputs = {
      {0.0, 0.0},
      {0.0, 1.0},
      {1.0, 0.0},
      {1.0, 1.0}
  };

  std::vector<std::vector<double>> xor_outputs = {
      {0.0},
      {1.0},
      {1.0},
      {0.0}
  };

  // Display XOR truth table
  std::cout << "XOR Truth Table:\n";
  std::cout << "Input -> Output\n";
  for (size_t i = 0; i < xor_inputs.size(); i++)
  {
    std::cout << "[" << xor_inputs[i][0] << ", " << xor_inputs[i][1] << "] -> " << xor_outputs[i][0] << "\n";
  }
  std::cout << "\n";

  // Create neural network
  Model* network = new Model(0.3);
  int nodes = 10;
  Layer* input_layer = network->CreateFullLayer(2, Layer::sigmoid);    
  Layer* hidden_layer = network->CreateFullLayer(nodes, Layer::sigmoid);  
  Layer* output_layer = network->CreateFullLayer(1, Layer::sigmoid);   

  network->SetDropRate(0.0);
  std::cout << "Starting training...\n\n";

  // Training parameters
  const int epochs = 8000;
  const int print_interval = 500;

  // Training loop
  for (int epoch = 0; epoch < epochs; epoch++) 
  {
    for (size_t i = 0; i < xor_inputs.size(); i++) 
    {
      network->LearnData(xor_inputs[i], xor_outputs[i]);
    }

    // Calculate average error for this epoch
    double total_error = network->GetTotalError() / xor_inputs.size();
    if ((epoch + 1) % 5000 == 0)
    {
      double initial_lr = network->GetLearningRate();
      double decay_rate = 0.95;
      double min_lr = 0.001;
      double current_lr = std::max(initial_lr * pow(decay_rate, epoch / 1000.0), min_lr);
    }
    if ((epoch + 1) % print_interval == 0)
    {
      std::cout << "Epoch [" << std::setw(4) << (epoch + 1) << "/" << epochs << "], Loss: " << std::fixed << std::setprecision(6) << total_error << "\n";
    }
  }

  std::cout << "\nTraining completed!\n\n";

  // Test the trained network
  std::cout << "Testing trained XOR gate:\n";
  std::cout << "Input -> Target -> Prediction -> Rounded\n";

  for (size_t i = 0; i < xor_inputs.size(); i++) 
  {
    network->InputData(xor_inputs[i]);
   
    // Get prediction from output layer
    std::vector<double> prediction = network->GetOutput();// output_layer->outputs[0];
    int rounded = (prediction.at(0) > 0.5) ? 1 : 0;

    std::cout << "[" << xor_inputs[i][0] << ", " << xor_inputs[i][1] << "] -> " << xor_outputs[i][0] << " -> " << std::fixed << std::setprecision(4) << prediction.at(0) << " -> " << rounded << "\n";
  }

  // Calculate accuracy
  int correct_predictions = 0;
  for (size_t i = 0; i < xor_inputs.size(); i++)
  {
    network->InputData(xor_inputs[i]);
   
    std::vector<double> prediction = network->GetOutput();
    int rounded = (prediction.at(0) > 0.5) ? 1 : 0;
    int target = static_cast<int>(xor_outputs[i][0]);

    if (rounded == target)
    {
      correct_predictions++;
    }
  }

  double accuracy = (static_cast<double>(correct_predictions) / xor_inputs.size()) * 100.0;
  std::cout << "\nAccuracy: " << std::fixed << std::setprecision(1) << accuracy << "%\n";

  // Test with custom inputs
  std::cout << "\nTesting with custom inputs:\n";
  std::vector<std::vector<double>> test_inputs = {
      {0.0, 0.0},
      {0.0, 1.0},
      {1.0, 0.0},
      {1.0, 1.0}
  };

  for (const auto& input : test_inputs) 
  {
    network->InputData(input);

    std::vector<double> output = network->GetOutput();
    int rounded = (output.at(0) > 0.5) ? 1 : 0;

    std::cout << "XOR(" << input[0] << ", " << input[1] << ") = " << std::fixed << std::setprecision(4) << output.at(0) << " ≈ " << rounded << "\n";
  }

  // Cleanup
  delete network;

  return 0;
}
