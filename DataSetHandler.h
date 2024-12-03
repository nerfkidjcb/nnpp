#ifndef DATASET_HANDLER_H
#define DATASET_HANDLER_H

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

class DataSetHandler {
public:
    // Constructor
    DataSetHandler(const std::vector<std::vector<double>>& inputs, 
                   const std::vector<std::vector<double>>& targets, 
                   double trainRatio = 0.8);

    // Function to split the data into training and validation sets
    void splitData();

    // Getter functions for the splits
    const std::vector<std::vector<double>>& getTrainInputs() const;
    const std::vector<std::vector<double>>& getTrainTargets() const;
    const std::vector<std::vector<double>>& getValInputs() const;
    const std::vector<std::vector<double>>& getValTargets() const;

    // Function to print the size of the splits (optional)
    void printSplitSizes() const;

private:
    // Data
    std::vector<std::vector<double>> inputs_;
    std::vector<std::vector<double>> targets_;

    // Split ratio for training data
    double trainRatio_;

    // Training and validation data
    std::vector<std::vector<double>> trainInputs_;
    std::vector<std::vector<double>> trainTargets_;
    std::vector<std::vector<double>> valInputs_;
    std::vector<std::vector<double>> valTargets_;
    
    // Helper function for shuffling and splitting the data
    void shuffleAndSplitData();
};

#endif  // DATASET_HANDLER_H
