# NNPP (Neural Network Plus Plus)
Silly manual implementation of nns in cpp

## What is here
This is mostly just an excerise in cpp for me but here:

- So far I've made a basic shallow net that can handle very little (4 feature straight lines and single feature quadratic). But its the thought that counts. 

- Also a dataset generator that can generate a regression dataset by linearly combining arbitrary number of features passed through a function. (Currently the output dimesnions is a lie and just is the same vector for each output dimension)
- And then a dataset handler that more or less just handles the splitting of the generated dataset into training and testing sets.

## Run it
If you are somehow reading this, just compile the working* shallow net with `g++ -o shallow_network main.cpp ShallowNeuralNetwork.cpp DataSetHandler.cpp DataSetGenerator.cpp` and run it with `./shallow_network`