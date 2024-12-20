#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "ShallowNeuralNetwork.h"
#include "DeepNeuralNetwork.h"
#include "DataSetHandler.h"
#include "DataSetGenerator.h"

int main() {
    DataSetGenerator generator(6, 1, 1000); // Input size, output size, number of examples. 4 inputs is basically as good as we get here for shallow

    // Define a function to generate polynomial sequences
    auto func = [](double x) { return x * 3 + 2; }; // mx + c is what this shallow guy is pretty ok at

    // Generate a dataset with noise
    auto [inputs, targets] = generator.generatePolynomials(func, true, 0.2); // Funtion, addNoise, noiseLevel

    // Print a few input-target pairs, no matter the dimension
    for (int i = 0; i < 5; ++i) {
        std::cout << "Input: ";
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            std::cout << inputs[i][j] << " ";  
        }

        std::cout << ", Target: ";
        for (size_t j = 0; j < targets[i].size(); ++j) {
            std::cout << targets[i][j] << " "; 
        }

        std::cout << std::endl;
    }



    // Initialize DataSetHandler to split data into training and validation sets
    DataSetHandler dataSetHandler(inputs, targets, 0.8);  // 80% for training, 20% for validation
    dataSetHandler.splitData();
    dataSetHandler.printSplitSizes();

    // Create a neural network instance
    // ShallowNeuralNetwork nn(inputs[0].size(), 100, targets[0].size());  // Input size, hidden size, output size
    DeepNeuralNetwork nn(inputs[0].size(), {5, 10, 5}, targets[0].size(), false);  // Input size, hidden sizes, output size, useMultithreading

    
    // TODO get the dnn working, and then create multithreaded methods for back prop 
    // also will need better gd methods as we are gonna start finding local minima (i think, i cant picture ND space) with the dnn

    // Train the neural network using the training data
    auto start = std::chrono::high_resolution_clock::now();

    nn.train(dataSetHandler.getTrainInputs(), dataSetHandler.getTrainTargets(), 100, 0.0001);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Training time: " << diff.count() << " s" << std::endl; // See what multithreading does to this

    // After training, test the network on the validation set
    std::vector<std::vector<double>> valInputs = dataSetHandler.getValInputs();
    std::vector<std::vector<double>> valTargets = dataSetHandler.getValTargets();

    float totalError = 0.0;
    int numValidationSamples = valInputs.size();

    for (int i = 0; i < numValidationSamples; ++i) {
        // Feedforward pass for each validation input
        DeepFeedforwardResult result = nn.feedforward(valInputs[i]);  // Get both hidden and output activations (for DeepNN)
        // FeedforwardResult result = nn.feedforward(valInputs[i]);  // Get both hidden and output activations (for ShallowNN)
        std::vector<double> output = result.output;  // Extract output activations

        float error = 0.0;

        // Calculate squared error for this sample
        for (int j = 0; j < output.size(); ++j) {
            error += (output[j] - valTargets[i][j]) * (output[j] - valTargets[i][j]);
        }
        
        totalError += error;  // Add to the total error
    }

    // Calculate average MSE (Mean Squared Error) for the validation set
    float avgError = totalError / numValidationSamples;

    // Print the average validation MSE
    std::cout << "Average validation MSE: " << avgError << std::endl;

    // Show some example predictions vs. actual targets
    std::cout << "Example predictions vs. actual targets:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        DeepFeedforwardResult result = nn.feedforward(valInputs[i]);  // Feedforward pass (for DeepNN)
        // FeedforwardResult result = nn.feedforward(valInputs[i]);  // Feedforward pass (for ShallowNN)
        std::vector<double> output = result.output;  // Extract output activations

        std::cout << "Prediction: " << output[0] << ", Target: " << valTargets[i][0] << std::endl;
    }


    return 0;
}
