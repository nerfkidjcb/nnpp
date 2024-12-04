# NNPP (Neural Network Plus Plus)
Silly manual implementation of nns in cpp

## What is here
This is mostly just an excerise in cpp for me but here:

- So far I've made a basic shallow net that can handle very little (4 feature straight lines and single feature quadratic). But its the thought that counts. 
- Also made a deep net implementation that can handle more than the shallow, and can the depth and width of the network can be defined.

- Also a dataset generator that can generate a regression dataset by linearly combining arbitrary number of features passed through a function. (Currently the output dimesnions is a lie and just is the same vector for each output dimension)
- And then a dataset handler that more or less just handles the splitting of the generated dataset into training and testing sets.

## Run it
Bare in mind that some tweaks to `main.cpp` might be needed to run the shallow net as its currently set up to run the deep net.

- Compile the deep net with `g++ -o deep_network main.cpp DeepNeuralNetwork.cpp DataSetHandler.cpp DataSetGenerator.cpp` and run it with `./deep_network`

- Or compile the shallow net with `g++ -o shallow_network main.cpp ShallowNeuralNetwork.cpp DataSetHandler.cpp DataSetGenerator.cpp` and run it with `./shallow_network`


