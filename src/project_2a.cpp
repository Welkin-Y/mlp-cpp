#include "dense_linear_algebra.h"
#include "project2_a.h"
#include "project2_a_basics.h"
#include <iostream>
#include <utility>

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <filename> <learning_rate> <tolerance> <max_iteration>"
              << std::endl;
    return 1;
  }
  const std::string file_name = argv[1];
  std::cout << "Using file: " << file_name << std::endl;

  double learning_rate = std::atof(argv[2]);
  double tol_training = std::atof(argv[3]); // Convergence tolerance
  unsigned max_iter = std::atoi(argv[4]);   // Max iterations
  //
  int tmp;
  std::vector<int> neurons;
  std::cout << "Enter the number of neurons in next layer, ^D to finish: ";
  while (std::cin >> tmp) {
    std::cout << "Enter the number of neurons in next layer, ^D to finish: ";
    neurons.emplace_back(tmp);
  }
  if (neurons.size() < 2) {
    std::cerr << "At least two layers are needed" << std::endl;
    return 1;
  }

  ///
  // Instantiate an activation function (same for all layers)
  //---------------------------------------------------------
  ActivationFunction *activation_function_pt = new TanhActivationFunction;

  // Build the network: 2,3,3,1 neurons in the four layers
  //------------------------------------------------------
  // Number of neurons in the input layer
  unsigned n_input = neurons[0];
  // Storage for the non-input layers: combine the number of neurons
  // and the (pointer to the) activation function into a pair
  // and store one pair for each layer in a vector:
  std::vector<std::pair<unsigned, ActivationFunction *>> non_input_layer;

  for (unsigned i = 1; i < neurons.size(); i++) {
    non_input_layer.emplace_back(
        std::make_pair(neurons[i], activation_function_pt));
  }

  NeuralNetwork nn(n_input, non_input_layer); // Input layer has 2 neurons

  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
  nn.read_training_data(file_name, training_data);

  nn.train(training_data, learning_rate, tol_training, max_iter);
}
