#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H

#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <iostream>

// Struct for generator results
struct GeneratorResult {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
};

class DataSetGenerator {
public:
    // Constructor to initialize parameters for sequence generation
    DataSetGenerator(int inputSize, int outputSize, int numExamples);

    // Function to generate polynomial sequences
    GeneratorResult generatePolynomials(const std::function<double(double)>& polyFunc, bool addNoise = false, double noiseLevel = 0.1);

    // Function to print dataset
    void printDataSet(const std::vector<std::vector<double>>& dataset);

private:
    int inputSize;  // Number of features (input dimension)
    int outputSize; // Number of outputs (target dimension)
    int numExamples; // Number of examples in the dataset

    // Utility function to generate random values
    double randomValue(double min, double max);
};

#endif // DATASETGENERATOR_H
