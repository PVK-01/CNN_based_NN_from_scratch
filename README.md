# CNN based NN from scratch
## Introduction, and Aim
The main aim of this repo is to create a convolution based neural network from scratch. This neural network created will have options of including batch norm for normal layers, layer norm for convolution layers, dropouts, the L1 or L2 normalization, intializing the weights with a given standard deviation value, and the option of choosing desired optimization method until ADAM.
The folder NNDL contains .py files for each aspect or layer building of the neural network.
The layers.py file has the building code for each individual layers like affine, RELu, batch norm, droupout layer, etc.
The optim.py file contains code for execution of the different optimization methods.
The conv_layers.py contains the code for the basic convolutional layers.
The conv_layer_utils.py contain the code for combining different convolutional layers into a single function.
Now using all these files and solver (not build by me), the convolutional neural network was built.
The .ipynb files shows about the utilization of these built methods.