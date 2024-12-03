#ifndef SHALLOWNEURALNETWORK_H
#define SHALLOWNEURALNETWORK_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Struct for hidden and output layer values
struct FeedforwardResult {
    std::vector<double> hidden;
    std::vector<double> output;
};

class ShallowNeuralNetwork {
public:
    // Constructor to initialize the network
    ShallowNeuralNetwork(int inputSize, int hiddenSize, int outputSize);

    // Public methods
    FeedforwardResult feedforward(const std::vector<double>& input);  // Now returns both hidden and output activations
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learningRate);

private:
    // Network parameters
    int inputSize, hiddenSize, outputSize;

    // Weights and biases
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasHidden;
    std::vector<double> biasOutput;

    // Helper methods for initialization and activation functions
    void initializeWeights();
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    double relu(double x);
    double reluDerivative(double x);
    double leakyRelu(double x);
    double leakyReluDerivative(double x);
};

#endif // SHALLOWNEURALNETWORK_H
