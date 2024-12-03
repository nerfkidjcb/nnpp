#include "DataSetGenerator.h"
#include <iostream>
#include <random>
#include <functional>
#include <cmath>

// Constructor to initialize parameters
DataSetGenerator::DataSetGenerator(int inputSize, int outputSize, int numExamples)
    : inputSize(inputSize), outputSize(outputSize), numExamples(numExamples) {}

// Utility function to generate random values between min and max
double DataSetGenerator::randomValue(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Function to generate polynomial sequences
GeneratorResult DataSetGenerator::generatePolynomials(
    const std::function<double(double)>& polyFunc, bool addNoise, double noiseLevel) {
    
    std::vector<std::vector<double>> inputs;   // To hold input vectors
    std::vector<std::vector<double>> targets;  // To hold target vectors

    for (int i = 0; i < numExamples; ++i) {
        std::vector<double> input(inputSize, 0.0);
        
        // Generate random inputs for the current example
        for (int j = 0; j < inputSize; ++j) {
            input[j] = randomValue(-10.0, 10.0);  // Random values between -10 and 10
        }

        // Calculate the output based on the polynomial function applied to the input
        std::vector<double> output(outputSize, 0.0);

        for (int j = 0; j < outputSize; ++j) {
            double sum = 0.0;

            // Apply the polynomial function to each input element and sum the results
            for (int i = 0; i < inputSize; ++i) {
                sum += polyFunc(input[i]);  // Apply polyFunc to each input[i] and accumulate the result ie y = x1^2 + x2^2 + ... + xn^2
            }

            output[j] = sum;
        } // TODO actually make the outputs different in some way, this atm essentially just generates identical values for each output dimension

        if (addNoise) {
            for (int j = 0; j < outputSize; ++j) {
                output[j] += randomValue(-noiseLevel, noiseLevel); 
            }
        }

        inputs.push_back(input);
        targets.push_back(output);
    }

    GeneratorResult result = {inputs, targets};
    return result;
}

// Function to print dataset, cos I miss python
void DataSetGenerator::printDataSet(const std::vector<std::vector<double>>& dataset) {
    for (const auto& data : dataset) {
        for (double val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}
