#ifndef DEEPNEURALNETWORK_H
#define DEEPNEURALNETWORK_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Struct for hidden and output layer values
struct DeepFeedforwardResult {
    std::vector<std::vector<double>> hidden; 
    std::vector<double> output;
};

class DeepNeuralNetwork {
public:
    // Constructor to initialize the network
    DeepNeuralNetwork(int inputSize, std::vector<int> hiddenSizes, int outputSize, bool useMultithreading = true);

    // Public methods
    DeepFeedforwardResult feedforward(const std::vector<double>& input);  // Now returns both hidden and output activations
    DeepFeedforwardResult feedforwardMultithreaded(const std::vector<double>& input);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learningRate);

private:
    // Network parameters
    int inputSize, outputSize;
    std::vector<int> hiddenSizes;
    bool useMultithreading = true;

    // Weights and biases
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;


    // Helper methods for initialization and activation functions
    void initializeWeights();
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    double relu(double x);
    double reluDerivative(double x);
    double leakyRelu(double x);
    double leakyReluDerivative(double x);
};

#endif // DEEPNEURALNETWORK_H
