# Prototype-Networks
This repository includes routines for constructing, deploying and testing prototype based neural networks of one or more layers. Examples.py includes demos of how tremor data can be loaded, pre-processed and classified using single or multi-layer prototypical neural network. The Prototypical_NN module contains the routines needed for formulating and training the proposed neural network modules in Evers et al. 2024. Utility functions for the examples are in utils.py. 

## Requirements
Vanila implementation of the methods relies on up-to-date Scipy and Numpy libraries, code was tested on Python 3.9. The radial basis function networks underlying the prototypical neural networks are closely linked to Gaussian processes and as such one can significantly optimize training of the methods using sprase Gaussian processes approximations such as inducing points. See Evers et al. 2024 for more details. To eneable such scalable inference, using the option ```training_method = GP-mode```
