#include "ShallowNeuralNetwork.h"
#include <iostream>

// Constructor to initialize the network
ShallowNeuralNetwork::ShallowNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
    initializeWeights();
}

// Initialize weights and biases with small random values
void ShallowNeuralNetwork::initializeWeights() {
    // Initialize input-hidden weights
    weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsInputHidden[i][j] = (rand() % 1000) / 1000.0 - 0.5; // Random values between -0.5 and 0.5
        }
    }

    // Initialize hidden-output weights
    weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weightsHiddenOutput[i][j] = (rand() % 1000) / 1000.0 - 0.5; // Random values between -0.5 and 0.5
        }
    }

    // Initialize biases
    biasHidden.resize(hiddenSize, 0.0);
    biasOutput.resize(outputSize, 0.0);
}

// Sigmoid activation function
double ShallowNeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid activation function
double ShallowNeuralNetwork::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// ReLU activation function
double ShallowNeuralNetwork::relu(double x) {
    return (x > 0) ? x : 0.0;
}

// Derivative of ReLU activation function
double ShallowNeuralNetwork::reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0; 
}

// Leaky ReLU activation function
double ShallowNeuralNetwork::leakyRelu(double x) {
    return (x > 0) ? x : 0.01 * x; // Leaky slope is 0.01, hardcoded here and in derivative
}

// Derivative of Leaky ReLU activation function
double ShallowNeuralNetwork::leakyReluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.01;
}

// Feedforward to get activations for both hidden and output layers
FeedforwardResult ShallowNeuralNetwork::feedforward(const std::vector<double>& input) {
    // Hidden layer activation
    std::vector<double> hidden(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            hidden[i] += input[j] * weightsInputHidden[j][i];
        }
        hidden[i] += biasHidden[i];
        hidden[i] = relu(hidden[i]); // Apply ReLU activation
    }

    // Output layer activation
    std::vector<double> output(outputSize, 0.0);
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            output[i] += hidden[j] * weightsHiddenOutput[j][i];
        }
        output[i] += biasOutput[i]; // Worth noting we didnt put an activation function here
    }

    FeedforwardResult result = {hidden, output};
    return result;
}

// Training the network with gradient descent
void ShallowNeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learningRate) {
    int numExamples = inputs.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        // Loop over each training example
        for (int i = 0; i < numExamples; ++i) {
            const std::vector<double>& input = inputs[i];
            const std::vector<double>& target = targets[i];

            FeedforwardResult result = feedforward(input);
            std::vector<double>& hidden = result.hidden;
            std::vector<double>& output = result.output;

            double loss = 0.0; // MSE
            for (int j = 0; j < outputSize; ++j) {
                loss += pow(output[j] - target[j], 2);
            }
            totalLoss += loss / outputSize;

            // Backpropagation
            // Calculate output layer error
            std::vector<double> outputError(outputSize, 0.0);
            for (int j = 0; j < outputSize; ++j) {
                outputError[j] = output[j] - target[j];
            }

            // Calculate hidden layer error
            std::vector<double> hiddenError(hiddenSize, 0.0);
            for (int j = 0; j < hiddenSize; ++j) {
                for (int k = 0; k < outputSize; ++k) {
                    hiddenError[j] += outputError[k] * weightsHiddenOutput[j][k];
                }
            }

            // Update weights and biases (gradient descent)
            // Output layer
            for (int j = 0; j < outputSize; ++j) {
                for (int k = 0; k < hiddenSize; ++k) {
                    weightsHiddenOutput[k][j] -= learningRate * outputError[j] * hidden[k];
                }
                biasOutput[j] -= learningRate * outputError[j];
            }

            // Hidden layer
            for (int j = 0; j < hiddenSize; ++j) {
                for (int k = 0; k < inputSize; ++k) {
                    weightsInputHidden[k][j] -= learningRate * hiddenError[j] * input[k] * reluDerivative(hidden[j]);
                }
                biasHidden[j] -= learningRate * hiddenError[j] * reluDerivative(hidden[j]);
            }
        }

        // Print loss every 10 epochs
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << totalLoss / numExamples << std::endl;
        }
    }
}
