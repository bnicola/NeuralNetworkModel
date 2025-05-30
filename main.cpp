#include "Model.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

// Generate XOR dataset
std::vector<std::pair<std::vector<double>, std::vector<double>>> generateXORData() 
{
  std::vector<std::pair<std::vector<double>, std::vector<double>>> data;

  // XOR truth table
  data.push_back({ {0.0, 0.0}, {0.0} });  // 0 XOR 0 = 0
  data.push_back({ {0.0, 1.0}, {1.0} });  // 0 XOR 1 = 1
  data.push_back({ {1.0, 0.0}, {1.0} });  // 1 XOR 0 = 1
  data.push_back({ {1.0, 1.0}, {0.0} });  // 1 XOR 1 = 0

  return data;
}

void testStableXOR(Model& model) 
{
   auto data = generateXORData();

  // Training with adaptive learning rate
  int epochs = 10000;  // Reduced epochs
 
  std::cout << "Training for " << epochs << " epochs with adaptive learning rate..." << std::endl;

  for (int epoch = 0; epoch < epochs; epoch++) 
  {
    for (auto& sample : data)
    {
      model.LearnData(sample.first, sample.second);
    }

    if (epoch % 500 == 0) 
    {
      double avgError = model.GetAverageError(data.size() * 500);
      std::cout << "Epoch " << epoch << ", Average Error: " << avgError  << ", LR: " << model.GetLearningRate() << std::endl;
    }
  }

  // Testing
  std::cout << "\nTesting XOR results:" << std::endl;
  std::cout << std::left << std::setw(15) << "Input" << std::setw(12) << "-> Output" << std::setw(12) << "(Expected)" << "Error" << std::endl;
  std::cout << std::setfill('-') << std::setw(50) << "" << std::setfill(' ') << std::endl;

  double totalTestError = 0.0;
  for (auto& sample : data) 
  {
    model.InputData(sample.first);
    auto output = model.GetOutput();
    double error = abs(output[0] - sample.second[0]);
    totalTestError += error;

    std::cout << "[" << sample.first[0] << ", " << sample.first[1] << "]\t-> " << std::fixed << std::setprecision(6) << output[0] << "\t(" << sample.second[0] << ")\t\t" << error << std::endl;
  }

  std::cout << "Average Test Error: " << totalTestError / data.size() << std::endl;
}

int main()
{
  bool trained = false;
  Model model3(0.001); 
  if (trained)
  {
    model3.LoadModel("deep");
    model3.SetDropRate(0.1);
    testStableXOR(model3);
    double totalTestError = 0.0;
    auto data = generateXORData();

    for (auto& sample : data)
    {
      model3.InputData(sample.first);
      auto output = model3.GetOutput();
      double error = abs(output[0] - sample.second[0]);
      totalTestError += error;

      std::cout << "[" << sample.first[0] << ", " << sample.first[1] << "]\t-> " << std::fixed << std::setprecision(6) << output[0] << "\t(" << sample.second[0] << ")\t\t" << error << std::endl;
    }
  }
  else
  {
    model3.SetDropRate(0.1);
    int numOfNodes = 40;

    model3.CreateInputLayer(1, 2, 1);
    
    for (int i = 0; i < 10; i++)
    {
      model3.CreateFullLayer(numOfNodes, Layer::ActivationFunc::tanh);
    }

    model3.CreateResidualFullLayer(numOfNodes, Layer::ActivationFunc::relu, 1);
    model3.CreateResidualFullLayer(numOfNodes, Layer::ActivationFunc::relu, 2);
    model3.CreateResidualFullLayer(numOfNodes, Layer::ActivationFunc::relu, 4);
    model3.CreateFullLayer(numOfNodes, Layer::ActivationFunc::tanh);
    model3.CreateFullLayer(1, Layer::ActivationFunc::tanh);  // Changed from sigmoid
    testStableXOR(model3);
    model3.SaveModel("deep");
  } 

  return 0;
}
