#include "DataSetHandler.h"
#include <random>
#include <algorithm>
#include <iostream>

// Constructor
DataSetHandler::DataSetHandler(const std::vector<std::vector<double>>& inputs, 
                               const std::vector<std::vector<double>>& targets, 
                               double trainRatio)
    : inputs_(inputs), targets_(targets), trainRatio_(trainRatio) {}

// Function to split the data
void DataSetHandler::splitData() {
    shuffleAndSplitData();
}

// Getter functions
const std::vector<std::vector<double>>& DataSetHandler::getTrainInputs() const {
    return trainInputs_;
}

const std::vector<std::vector<double>>& DataSetHandler::getTrainTargets() const {
    return trainTargets_;
}

const std::vector<std::vector<double>>& DataSetHandler::getValInputs() const {
    return valInputs_;
}

const std::vector<std::vector<double>>& DataSetHandler::getValTargets() const {
    return valTargets_;
}

// Print the size of the training and validation sets
void DataSetHandler::printSplitSizes() const {
    std::cout << "Training data size: " << trainInputs_.size() << std::endl;
    std::cout << "Validation data size: " << valInputs_.size() << std::endl;
}

// Helper function for shuffling and splitting data
void DataSetHandler::shuffleAndSplitData() {
    // Get the total number of data points
    size_t dataSize = inputs_.size();
    size_t trainSize = static_cast<size_t>(dataSize * trainRatio_);
    
    // Create a vector of indices
    std::vector<int> indices(dataSize);
    for (int i = 0; i < dataSize; ++i) {
        indices[i] = i;
    }
    
    // Shuffle the indices randomly using a random device
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Split the data into training and validation sets based on the shuffled indices
    for (int i = 0; i < dataSize; ++i) {
        int idx = indices[i];
        if (i < trainSize) {
            // Add to training set
            trainInputs_.push_back(inputs_[idx]);
            trainTargets_.push_back(targets_[idx]);
        } else {
            // Add to validation set
            valInputs_.push_back(inputs_[idx]);
            valTargets_.push_back(targets_[idx]);
        }
    }
}
