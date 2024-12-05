#include "DeepNeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <thread>

// Constructor to initialize the network
DeepNeuralNetwork::DeepNeuralNetwork(int inputSize, std::vector<int> hiddenSizes, int outputSize, bool useMultithreading)
    : inputSize(inputSize), hiddenSizes(hiddenSizes), outputSize(outputSize), useMultithreading(useMultithreading) {
    initializeWeights();
}

// Initialize weights and biases with small random values
void DeepNeuralNetwork::initializeWeights() {
    // Initialize weights and biases for each layer
    weights.resize(hiddenSizes.size() + 1);  // +1 for the output layer
    biases.resize(hiddenSizes.size() + 1);   // +1 for the output layer

    // Initialize input to first hidden layer
    weights[0].resize(inputSize, std::vector<double>(hiddenSizes[0]));
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSizes[0]; ++j) {
            weights[0][i][j] = (rand() % 1000) / 1000.0 - 0.5; // Random values between -0.5 and 0.5
        }
    }
    biases[0].resize(hiddenSizes[0], 0.0);

    // Initialize hidden layers
    for (int l = 1; l < hiddenSizes.size(); ++l) {
        weights[l].resize(hiddenSizes[l-1], std::vector<double>(hiddenSizes[l]));
        for (int i = 0; i < hiddenSizes[l-1]; ++i) {
            for (int j = 0; j < hiddenSizes[l]; ++j) {
                weights[l][i][j] = (rand() % 1000) / 1000.0 - 0.5; // Random values between -0.5 and 0.5
            }
        }
        biases[l].resize(hiddenSizes[l], 0.0);
    }

    // Initialize hidden to output layer
    weights[hiddenSizes.size()].resize(hiddenSizes.back(), std::vector<double>(outputSize));
    for (int i = 0; i < hiddenSizes.back(); ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weights[hiddenSizes.size()][i][j] = (rand() % 1000) / 1000.0 - 0.5; // Random values between -0.5 and 0.5
        }
    }
    biases[hiddenSizes.size()].resize(outputSize, 0.0);

    // Optionally print what the weights look like if you fancy this. Should follow the whole inputxhidden1width, hidden1widthxhidden2width, hidden2widthxoutputwidth (if you have 2 hidden layers)
    // for (int i = 0; i < weights.size(); ++i) {
    //     std::cout << "Layer " << i << std::endl;
    //     for (int j = 0; j < weights[i].size(); ++j) {
    //         for (int k = 0; k < weights[i][j].size(); ++k) {
    //             std::cout << weights[i][j][k] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
}


// Sigmoid activation function
double DeepNeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid activation function
double DeepNeuralNetwork::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// ReLU activation function
double DeepNeuralNetwork::relu(double x) {
    return (x > 0) ? x : 0.0;
}

// Derivative of ReLU activation function
double DeepNeuralNetwork::reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// Leaky ReLU activation function
double DeepNeuralNetwork::leakyRelu(double x) {
    return (x > 0) ? x : 0.01 * x; // Leaky slope is 0.01, hardcoded here and in derivative
}

// Derivative of Leaky ReLU activation function
double DeepNeuralNetwork::leakyReluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.01;
}

// Multithreaded feedforward method for hidden layer activations

DeepFeedforwardResult DeepNeuralNetwork::feedforwardMultithreaded(const std::vector<double>& input) {
    // This vector holds the activations of the hidden layers
    std::vector<std::vector<double>> hiddenActivations(hiddenSizes.size());
    std::vector<double> output(outputSize, 0.0);
    
    // Input to the first hidden layer (and then passed sequentially to each subsequent hidden layer)
    std::vector<double> inputToHidden = input;

    // Loop over each layer
    for (size_t layer = 0; layer < hiddenSizes.size(); ++layer) {
        std::vector<double> hidden(hiddenSizes[layer], 0.0);

        // Number of threads we want (24)
        size_t numThreads = 24;

        // Calculate the number of neurons each thread will handle
        size_t neuronsPerThread = (hiddenSizes[layer] + numThreads - 1) / numThreads;

        // Create a vector to hold the threads
        std::vector<std::thread> threads;

        // Launch threads
        for (size_t t = 0; t < numThreads; ++t) {
            threads.push_back(std::thread([&, t]() {
                // Each thread computes a chunk of neurons
                size_t startIdx = t * neuronsPerThread;
                size_t endIdx = (startIdx + neuronsPerThread < hiddenSizes[layer]) ? (startIdx + neuronsPerThread) : hiddenSizes[layer];                

                // Perform the calculation for this chunk of neurons
                for (size_t i = startIdx; i < endIdx; ++i) {
                    hidden[i] = 0.0;
                    for (size_t j = 0; j < inputToHidden.size(); ++j) {
                        hidden[i] += inputToHidden[j] * weights[layer][j][i];
                    }
                    hidden[i] += biases[layer][i];
                    hidden[i] = relu(hidden[i]); // Apply ReLU activation
                }
            }));
        }

        // Join all threads after they complete their work
        for (auto& t : threads) {
            t.join();
        }

        hiddenActivations[layer] = hidden;
        inputToHidden = hidden; // Set input to the current hidden layer's output
    }

    // Hidden to output layer (sequential, since output size is often small)
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSizes.back(); ++j) {
            output[i] += inputToHidden[j] * weights[hiddenSizes.size()][j][i];
        }
        output[i] += biases[hiddenSizes.size()][i];
    }

    DeepFeedforwardResult result = {hiddenActivations, output};
    return result;
}

// Feedforward to get activations for all layers
DeepFeedforwardResult DeepNeuralNetwork::feedforward(const std::vector<double>& input) {
    std::vector<std::vector<double>> hiddenActivations(hiddenSizes.size());
    std::vector<double> output(outputSize, 0.0);

    // Input to the first hidden layer (and then passed sequentially to each subsequent hidden layer)
    std::vector<double> inputToHidden = input;
    for (size_t layer = 0; layer < hiddenSizes.size(); ++layer) {
        std::vector<double> hidden(hiddenSizes[layer], 0.0);
        for (int i = 0; i < hiddenSizes[layer]; ++i) {
            for (size_t j = 0; j < inputToHidden.size(); ++j) {
                hidden[i] += inputToHidden[j] * weights[layer][j][i];
            }
            hidden[i] += biases[layer][i];
            hidden[i] = relu(hidden[i]); // Apply ReLU activation for now
        }
        hiddenActivations[layer] = hidden;
        inputToHidden = hidden; // Set input to the current hidden layer's output
    }

    // Hidden to output layer
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSizes.back(); ++j) {
            output[i] += inputToHidden[j] * weights[hiddenSizes.size()][j][i];
        }
        output[i] += biases[hiddenSizes.size()][i];
    }

    DeepFeedforwardResult result = {hiddenActivations, output};
    return result;
}

// Training the network with gradient descent
void DeepNeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learningRate) {
    int numExamples = inputs.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        // Loop over each training example
        for (int i = 0; i < numExamples; ++i) {
            const std::vector<double>& input = inputs[i];
            const std::vector<double>& target = targets[i];
        
            DeepFeedforwardResult result;
            
            if (useMultithreading) {  
                result = feedforwardMultithreaded(input); // seg faults yay
            } else {
                result = feedforward(input);
            }
            
            // Extract hiddenActivations and output references from the result
            std::vector<std::vector<double>>& hiddenActivations = result.hidden;
            std::vector<double>& output = result.output;


            double loss = 0.0; // Mean Squared Error
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

            // Calculate error for each hidden layer
            std::vector<std::vector<double>> hiddenErrors(hiddenSizes.size());
            for (int layer = hiddenSizes.size() - 1; layer >= 0; --layer) {
                std::vector<double> layerError(hiddenSizes[layer], 0.0);
                if (layer == hiddenSizes.size() - 1) { // Output layer
                    for (int j = 0; j < outputSize; ++j) {
                        for (int k = 0; k < hiddenSizes[layer]; ++k) {
                            layerError[k] += outputError[j] * weights[layer + 1][k][j];
                        }
                    }
                } else { // Hidden layers
                    for (int j = 0; j < hiddenSizes[layer]; ++j) {
                        for (int k = 0; k < hiddenSizes[layer + 1]; ++k) {
                            layerError[j] += hiddenErrors[layer + 1][k] * weights[layer + 1][j][k];
                        }
                    }
                }
                hiddenErrors[layer] = layerError;
            }

            // Update weights and biases using gradient descent
            // Output layer
            // Update output layer biases
            for (int j = 0; j < outputSize; ++j) {
                for (int k = 0; k < hiddenSizes.back(); ++k) {
                    weights[hiddenSizes.size()][k][j] -= learningRate * outputError[j] * hiddenActivations.back()[k];
                }
                biases[hiddenSizes.size()][j] -= learningRate * outputError[j];  // Corrected from `bias` to `biases`
            }

            // Update hidden layers biases
            for (int layer = hiddenSizes.size() - 1; layer >= 0; --layer) {
                for (int j = 0; j < hiddenSizes[layer]; ++j) {
                    for (int k = 0; k < (layer == 0 ? inputSize : hiddenSizes[layer - 1]); ++k) {
                        weights[layer][k][j] -= learningRate * hiddenErrors[layer][j] * (layer == 0 ? input[k] : hiddenActivations[layer - 1][k]) * reluDerivative(hiddenActivations[layer][j]);
                    }
                    biases[layer][j] -= learningRate * hiddenErrors[layer][j];  // Corrected from `bias` to `biases`
                }
            }

        }

        // Print loss every 10 epochs
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << totalLoss / numExamples << std::endl;
        }
    }
}
