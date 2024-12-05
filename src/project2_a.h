#include "dense_linear_algebra.h"
#include "project2_a_basics.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <random>

class NeuralNetworkLayer {
public:
  DoubleMatrix weights;
  DoubleVector biases;
  DoubleVector error;
  std::shared_ptr<DoubleVector> output;
  ActivationFunction *activationFunction;

  NeuralNetworkLayer(const unsigned int inDim, const unsigned int outDim,
                     ActivationFunction *activationFunc)
      : weights(inDim, outDim), biases(outDim), error(outDim),
        activationFunction(activationFunc) {
    output = std::make_shared<DoubleVector>(outDim);
  }

  unsigned get_input_dim() const { return weights.n(); }
  unsigned get_output_dim() const { return weights.m(); }

  /// Evaluate the feed-forward algorithm, running through the entire network,
  /// for input x. Feed it through the non-input-layers and return the
  /// output from the final layer.
  void computeOutput(const DoubleVector &input, DoubleVector &output) const {
    assert((void("Input vector has wrong size"), input.n() == get_input_dim()));
    assert(
        (void("Output vector has wrong size"), output.n() == get_output_dim()));
    // Compute weighted sum + bias for each neuron
    for (unsigned j = 0; j < output.n(); ++j) {
      double sum = 0.0;
      for (size_t i = 0; i < input.n(); ++i) {
        sum += input[i] * weights(i, j); // Weights(j, i) corresponds to the
                                         // weight for input[j] to neuron[i]
      }
      sum += biases[j]; // Add the bias
      output[j] = activationFunction->sigma(sum);
      this->output->operator[](j) = output[j];
    }
  }
};

class NeuralNetwork : NeuralNetworkBasis {
public:
  std::vector<NeuralNetworkLayer> layers;
  /// Constructor: Pass the number of inputs (i.e. the number of
  /// neurons in the input layer), n_input , and a vector of pairs ,
  /// containing for each subsequent layer (incl. the output layer)
  /// (i) the number of neurons in that layer
  /// (ii) a pointer to the activation function to be used by
  /// all neurons in that layer ,
  /// so:
  ///
  /// non_input_layer[l].first = number of neurons in non-input
  /// layer l
  ///
  /// non_input_layer[l].second = pointer to activation function
  /// for all neurons in non-input
  /// layer l
  ///
  /// Here l=0,1,..., with l=0 corresponding to the first internal
  /// (hidden) layer.
  NeuralNetwork(const unsigned &n_input,
                const std::vector<std::pair<unsigned, ActivationFunction *>>
                    &non_input_layer) {
    NeuralNetworkLayer new_layer(n_input, non_input_layer[0].first,
                                 non_input_layer[0].second);
    layers.push_back(new_layer);
    for (unsigned i = 1; i < non_input_layer.size(); i++) {
      NeuralNetworkLayer new_layer(non_input_layer[i - 1].first,
                                   non_input_layer[i].first,
                                   non_input_layer[i].second);
      initialise_parameters(0, 0.1, new_layer.weights, new_layer.biases);
      layers.push_back(new_layer);
    }
  }

  /// Evaluate the feed-forward algorithm, running through the entire network,
  /// for input x. Feed it through the non-input-layers and return the
  /// output from the final layer.
  virtual void feed_forward(const DoubleVector &input,
                            DoubleVector &output) const override {
    DoubleVector current_output = input; // Start with the input vector
    // Iterate over all layers

    for (unsigned i = 0; i < layers.size(); i++) {
      // Output of the current layer is stored in current_output
      output = DoubleVector(layers[i].get_output_dim());
      layers[i].computeOutput(current_output, output);

      // The output of this layer becomes the input for the next layer
      current_output = output;
    }
  }

  /// Get cost for given input and specified target output, so we're doing
  /// a single run through the network and compare the output to the target
  /// output.
  virtual double cost(const DoubleVector &input,
                      const DoubleVector &target_output) const override {
    double total_cost = 0.0;
    DoubleVector output;
    feed_forward(input, output);
    for (size_t i = 0; i < target_output.n(); ++i) {
      double error = output[i] - target_output[i];
      total_cost += error * error;
    }
    // Return the square error
    return total_cost;
  }

  /// Get cost for training data. The vector contains the training data
  /// in the form of pairs comprising
  ///   training_data[i].first  = input (a DoubleVector)
  ///   training_data[i].second = target_output (a DoubleVector)
  virtual double cost_for_training_data(
      const std::vector<std::pair<DoubleVector, DoubleVector>> training_data)
      const override {
    double ret{0};
    for (const auto &data_pair : training_data) {
      ret += cost(data_pair.first, data_pair.second);
    }
    return ret;
  }

  /// Get cost for ///    number of neurons in previous layer (unsigned)
  ///    number of neurons in current layer
  ///    0 b[0] (index and entry in bias vector)
  ///     :
  ///    n_neuron-1 b[n_neuron-1]
  ///    0 0 a[0][0] (indices and entry in weight matrix)
  ///    0 1 a[0][1]
  ///     :
  ///    1 0 a[1][0]
  ///    1 1 a[1][1]
  ///     :
  virtual void
  write_parameters_to_disk(const std::string &filename) const override {
    std::ofstream outFile(filename);
    // Ensure file is opened correctly
    if (!outFile) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }

    // Iterate over each non-input layer in the network
    for (size_t layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
      const NeuralNetworkLayer &layer = layers[layerIdx];

      // Write the name of the activation function
      outFile << layer.activationFunction->name() << std::endl;

      // Write the dimension m of the input (previous layer size)
      size_t m = layer.weights.m();
      outFile << m << std::endl;

      // Write the number of neurons n in the current layer
      size_t n = layer.weights.n();
      outFile << n << std::endl;

      // Write the bias vector (n lines, each containing two numbers)
      for (size_t j = 0; j < layer.biases.n(); ++j) {
        outFile << j << " " << layer.biases[j] << std::endl;
      }

      // Write the weight matrix (n x m lines, each containing three numbers)
      for (size_t i = 0; i < layer.weights.n(); ++i) {
        for (size_t j = 0; j < layer.weights.m(); ++j) {
          outFile << i << " " << j << " " << layer.weights(i, j) << std::endl;
        }
      }
    }

    // Close the file
    outFile.close();
  }

  /// Read in parameters for network (e.g. from previously
  /// performed training). See write_parameters_to_disk(...) for format
  virtual void read_parameters_from_disk(const std::string &filename) override {
    std::ifstream inFile(filename);

    // Ensure file is opened correctly
    if (!inFile) {
      std::cerr << "Error opening file for reading: " << filename << std::endl;
      return;
    }

    // Iterate over each non-input layer in the network
    for (size_t layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
      NeuralNetworkLayer &layer = layers[layerIdx];

      // Read the activation function name
      std::string activationFunctionName;
      std::getline(inFile, activationFunctionName);

      // Read the dimension m of the input (previous layer size)
      size_t m;
      inFile >> m;
      inFile.ignore(); // to ignore the newline

      // Read the number of neurons n in the current layer
      size_t n;
      inFile >> n;
      inFile.ignore(); // to ignore the newline

      // Read the bias vector (n lines, each containing two numbers)
      for (size_t j = 0; j < n; ++j) {
        size_t index;
        double value;
        inFile >> index >> value;
        inFile.ignore(); // to ignore the newline
        layer.biases[index] = value;
      }

      // Read the weight matrix (n x m lines, each containing three numbers)
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
          size_t row, col;
          double value;
          inFile >> row >> col >> value;
          inFile.ignore(); // to ignore the newline
          layer.weights(row, col) = value;
        }
      }
    }

    // Close the file
    inFile.close();
  }

  /// Train the network: specify filename containing training data
  /// in the form
  ///
  ///     training_data[i].first  = input
  ///     training_data[i].second = target output
  ///
  /// as well as the learning rate, the convergence tolerance and
  /// the max. number of iterations. Optional final argument: filename
  /// of the file used to document the convergence history.
  /// No convergence history is written if string is empty or not
  /// provided.
  virtual void
  train(const std::vector<std::pair<DoubleVector, DoubleVector>> &training_data,
        const double &learning_rate, const double &tol_training,
        const unsigned &max_iter,
        const std::string &convergence_history_file_name = "./log") override {
    // Initialize the convergence history vector (for tracking the cost)
    std::vector<double> convergence_history;

    // Iterate over the number of training iterations
    for (unsigned iter = 0; iter < max_iter; ++iter) {
      double total_cost = 0.0;

      // Shuffle the training data to ensure stochastic gradient descent
      // is random and not stuck in local minima
      std::random_device rd; // Obtain a random seed
      std::mt19937 g(rd());
      auto shuffled_data = training_data;
      std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);

      // Loop over each training example
      for (const auto &data_pair : shuffled_data) {
        const DoubleVector &input = data_pair.first;
        const DoubleVector &target_output = data_pair.second;

        // Feed-forward: Compute the output of the network for this input
        DoubleVector output(target_output.n());
        feed_forward(input, output);

        // Backpropagation: Calculate the error and update weights and biases
        // First, compute the error at the output layer
        DoubleVector output_error = DoubleVector(target_output.n());
        for (size_t i = 0; i < target_output.n(); ++i) {
          output_error[i] = (output[i] - target_output[i]) *
                            layers.back().activationFunction->dsigma(output[i]);
        }
        layers.back().error = output_error;

        // Hidden layers error propagation
        for (int layer_idx = layers.size() - 2; layer_idx >= 0; --layer_idx) {
          auto &layer = layers[layer_idx];
          auto &next_layer = layers[layer_idx + 1];

          // Compute the error for the current layer
          for (unsigned i = 0; i < layer.get_output_dim(); ++i) {
            double error_sum = 0.0;
            for (unsigned j = 0; j < next_layer.get_output_dim(); ++j) {
              error_sum += next_layer.weights(i, j) * next_layer.error[j];
            }
            layer.error[i] = error_sum * layer.activationFunction->dsigma(
                                             (*layer.output)[i]);
          }
        }
        // Update weights and biases using gradient descent
        for (unsigned layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
          auto &layer = layers[layer_idx];

          // Update weights
          for (unsigned i = 0; i < layer.get_output_dim(); ++i) {
            for (unsigned j = 0; j < layer.get_input_dim(); ++j) {
              double gradient =
                  layer.error[i] * (layer_idx == 0
                                        ? input[j]
                                        : (*layers[layer_idx - 1].output)[j]);
              layer.weights(j, i) -= learning_rate * gradient;
            }
          }

          // Update biases
          for (unsigned i = 0; i < layer.biases.n(); ++i) {
            layer.biases[i] -= learning_rate * layer.error[i];
          }
        }

        // Calculate the cost for this input-output pair and accumulate it
        total_cost += cost(input, target_output);
      }

      // Average the total cost across all training examples
      total_cost /= training_data.size();

      std::cout << "Iteration " << iter << " with cost " << total_cost
                << std::endl;

      // Check for convergence based on the tolerance
      if (iter > 0 && std::abs(total_cost) < tol_training) {
        std::cout << "Converged at iteration " << iter << " with cost "
                  << total_cost << std::endl;
        break;
      }

      // Log the cost for this iteration (for convergence tracking)
      convergence_history.push_back(total_cost);

      // Optionally, write convergence history to a file
      if (!convergence_history_file_name.empty()) {
        std::ofstream outFile(convergence_history_file_name, std::ios::app);
        if (outFile) {
          outFile << iter << " " << total_cost << std::endl;
        }
      }
    }

    // Optionally, save parameters to disk after training
    if (!convergence_history_file_name.empty()) {
      write_parameters_to_disk(convergence_history_file_name);
    }
  }

  /// Initialise parameters (weights and biases), drawing random numbers
  /// from a normal distribution with specified mean and standard
  /// deviation. This function is broken but demonstrates how to draw
  /// normally distributed random numbers. Please reimplement it so that
  /// the random values are assigned to your weights and biases.
  virtual void initialise_parameters(const double &mean, const double &std_dev,
                                     DoubleMatrix &weights,
                                     DoubleVector &biases) override {
    // Set up a normal distribution.

    // Calling "normal_dist(rnd)" then returns a random number drawn
    // from this distribution
    std::normal_distribution<> normal_dist(mean, std_dev);

    // Iterate over each layer's weights and biases to assign random values
    initRandomVector(biases, normal_dist);
    initRandomMatrix(weights, normal_dist);

    // throw std::runtime_error(
    //     "Never get here! Please overload this function\n");
  }

  // Helper functions
  //------------------

  /// Read training data from disk. Format:
  ///
  /// n_training_sets
  /// n_input
  /// n_output
  /// x[1], x[2],... x[n_input] y[1], y[2],... y[n_output]
  ///                    :
  ///     n_training_sets lines of data in total
  ///                    :
  /// x[1], x[2],... x[n_input] y[1], y[2],... y[n_output]
  /// end_of_file (literal string)
  ///
  virtual void
  read_training_data(const std::string &filename,
                     std::vector<std::pair<DoubleVector, DoubleVector>>
                         &training_data) const override {
    // TODO!
    // Wipe training data so we can push back
    training_data.clear();

    // Read parameters
    std::ifstream training_data_file(filename.c_str());
    unsigned n_training_sets = 0;
    training_data_file >> n_training_sets;
    unsigned n_input = 0;
    training_data_file >> n_input;
    unsigned n_output = 0;
    training_data_file >> n_output;

    // Read the actual data
    DoubleVector input(n_input);
    DoubleVector output(n_output);
    for (unsigned i = 0; i < n_training_sets; i++) {
      for (unsigned j = 0; j < n_input; j++) {
        training_data_file >> input[j];
      }
      for (unsigned j = 0; j < n_output; j++) {
        training_data_file >> output[j];
      }
      training_data.push_back(std::make_pair(input, output));
    }

    // Have we reached the end?
    std::string test_string;
    training_data_file >> test_string;
    if (test_string != "end_of_file") {
      throw std::runtime_error("\n\nERROR: Training set data doesn't end with "
                               "\"end_of_file\"\n\n");
    } else {
      std::cout << "Yay! Succesfully read training data in\n\n"
                << "        " << filename << "\n"
                << std::endl;
    }
  }
};
