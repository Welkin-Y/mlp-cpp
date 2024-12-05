#include "dense_linear_algebra.h"
#include "project2_a.h"
#include "project2_a_basics.h"
#include <iostream>
#include <utility>

int main() {
  ///
  // Instantiate an activation function (same for all layers)
  //---------------------------------------------------------
  ActivationFunction *activation_function_pt = new TanhActivationFunction;

  // Build the network: 2,3,3,1 neurons in the four layers
  //------------------------------------------------------
  // Number of neurons in the input layer
  unsigned n_input = 2;
  // Storage for the non-input layers: combine the number of neurons
  // and the (pointer to the) activation function into a pair
  // and store one pair for each layer in a vector:
  std::vector<std::pair<unsigned, ActivationFunction *>> non_input_layer;

  // The first internal (hidden) layer has 3 neurons
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // The second internal (hidden) layer has 3 neurons too!
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // Here's the output layer: A single neuron
  non_input_layer.push_back(std::make_pair(1, activation_function_pt));

  NeuralNetwork nn(n_input, non_input_layer); // Input layer has 2 neurons

  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
  const std::string file_name = "./project_training_data.dat";
  nn.read_training_data(file_name, training_data);

  double learning_rate = 0.1;
  double tol_training = 1e-4; // Convergence tolerance
  unsigned max_iter = 100000; // Max iterations

  nn.train(training_data, learning_rate, tol_training, max_iter);
}
