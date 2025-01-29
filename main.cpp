#include "Model.hpp"
#include <iostream>
#include <vector>
#include <array>

int main()
{
   Model model(0.0001);

   std::string modelName = "MyModel";
   bool myModel = false;// model.LoadModel(modelName);
   if (!myModel)
   {
      model.CreateFullLayer(2, Layer::leakyRelu);
      
      model.CreateFullLayer(25, Layer::tanh);
      model.CreateNormalizationLayer(25);
      model.CreateFullLayer(75, Layer::leakyRelu);
      model.CreateFullLayer(2, Layer::leakyRelu);

      //model.SaveModel(modelName);
   }
   model.SetDropRate(0);

   const std::array<std::array<double, 2>, 4> inputs = { {
       {0.01, 0.01},
       {0.01, 0.99},
       {0.99, 0.01},
       {0.99, 0.99}
   } };
   const std::array<std::array<double, 2>, 4> outputs = { {
       {7.1, 3.5},
       {8.2, 4.1},
       {4.3, -2.7},
       {-129.4, -3164.7}
   } };

   std::vector<double> in(2);
   std::vector<double> out(2);

   for (int i = 0; i < 15000; i++)
   {
      for (size_t j = 0; j < inputs.size(); ++j)
      {
         in[0] = inputs[j][0];
         in[1] = inputs[j][1];
         out[0] = outputs[j][0];
         out[1] = outputs[j][1];
         model.LearnData(in, out);
      }
   }

   model.SaveWeights();

   std::vector<double> output(2);
   for (const auto& input : inputs)
   {
      in[0] = input[0];
      in[1] = input[1];
      model.InputData(in);
      output = model.GetOutput();
      printf("in1= %f, in2= %f  ==> output1 = %f, output2 = %f\n", in[0], in[1], output[0], output[1]);
   }

   return 0;
}
