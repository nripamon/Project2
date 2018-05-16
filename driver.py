import Module_base as M
import numpy as np
import torch
from collections import OrderedDict
from opti import *
import initalization as init 
"""
Test Problem:
We use NN for the classification problem consisting in, given a certain
line, identyfing if a given point is above or below the line.
The structure of the model is:
- Single Linear layer,
- Tanh function,
and as Loss function we consider the MSE error.
"""

"""
Set the dimensions of the artificial train and test sets.
"""
# nb_train_input = 1000
# nb_test_input = 400

"""
Initialize train and test sets
"""
option = 2 # 0 for linear and 1 for circle and 2 for circle "projected" 
train_input, train_target, test_input, test_target, number_of_inputs, number_of_classes, nb_train_input, nb_test_input = init.initialize_dataset(option=option)


"""
Create List of Modules.
This list is then given as input to the constructor of the NN.
A loss function has to be provided separately.
"""
hidden_layer_1, hidden_layer_2, hidden_layer_3 = 30, 30, 30
# 1) Define the modules that are used in the network.
#.   N.B. Each operator that we are goind to use in the network is
#.   considered as a module and hence has to be defined.
Linear_1 = M.Linear(hidden_layer_1,number_of_inputs)
Nonlinear_1 = M.Tanh()
Linear_2 = M.Linear(hidden_layer_2,hidden_layer_1)
Nonlinear_2 = M.Tanh()
Linear_3 = M.Linear(number_of_classes,hidden_layer_2)
# 2) The modules are collected in dictionary.
operators = OrderedDict({ 1 : Linear_1,
                          2 : Nonlinear_1,
                          3 : Linear_2,
                          4 : Nonlinear_2,
                          5 : Linear_3 })
# 3) Define a dictionary representing the relations between
#.   modules. In this case we have the very simple structure:
#.        input --> Linear_1 --> Nonlinear_1 --> output
#    In the dictionary, for each module that do not produce the
#.   output, we must provide, in form of list, the set of connected 
#.   modules. 
#.   N.B. The values must be given in form of a list: the reason 
#.   behind this constraint is that other containers are not 
#    easily iterable.
connectivity = OrderedDict({ 1 : [2],
                             2 : [3],
                             3 : [4],
                             4 : [5]})
# 4) We specify the dictionary keys corresponding to the (possibly)
#.   multiple inputs and the output of the network.
#.   TODO: write a function that, given a dictionary, understand 
#.   automatically the positions of inputs and outputs.
input_operators = [ 1 ]
output_operator = 5



# hidden_layer_1 = 20
# hidden_1 = 40
# Linear_1 = M.Linear(number_of_classes,number_of_inputs)
# Nonlinear_1 = M.Tanh()
# operators = OrderedDict({ 1 : Linear_1,
#                          2 : Nonlinear_1 })
# connectivity = OrderedDict({ 1 : [2] })
# input_operators = [ 1 ]
# output_operator = 2



eta = 0.05



"""
Define the loss function and initialize the neural network.
"""
loss = M.LossMSE()
opt = SGD(eta=eta)
NN = M.Sequential(operators,connectivity,input_operators,output_operator,loss,opt)

"""
Set the number of epochs
"""
nb_epochs = 1000
mini_batch_size_inital = int(nb_train_input*0.05)
timestep = 1 # how many times we print the forward full batch





for k in range(0, nb_epochs):
    """
    During each epoch we have the following steps:
    1) Do the forward pass using the training set.
    2) Compute the related loss function.
    3) Do the backward pass to compute the gradients 
       wrt the weights and biases.
    4) Update the weights and biases.
    """
    mini_batch_size = mini_batch_size_inital
    shuffle_indexes_minibatch = torch.randperm(train_input.size(0))
    train_input = train_input[shuffle_indexes_minibatch]
    train_target = train_target[shuffle_indexes_minibatch] # question - why not long?

    for e in range(0, nb_train_input, mini_batch_size):

      mini_batch_size = min(mini_batch_size, nb_train_input - e)
      train_input_minibatch = train_input.narrow(0, e, mini_batch_size)
      train_target_minibatch = train_target.narrow(0, e, mini_batch_size)

      train_prediction = NN.forward(train_input_minibatch) 
      loss_output = NN.loss.forward(train_prediction,train_target_minibatch)
      NN.backward(train_prediction)
      NN.update_param()
    """
    After the training, we compute the number of 
    samples, in the training, that have been correctly
    predicted
    # TODO replace the following lines with a function.
    """
    #nb_train_errors = sum(abs(train_target_minibatch.max(1)[1] - train_prediction.max(1)[1]))
    
    """
    Print on terminal the loss function and the error percentages
    for the prediction for train and test sets
    """
    if np.mod(k,timestep)==0:

      train_prediction_batch = NN.forward(train_input) 
      loss_output = NN.loss.forward(train_prediction_batch,train_target)

      # nb_train_errors = sum(abs(train_target.max(1)[1] - train_prediction_batch.max(1)[1]))
      nb_train_errors = np.linalg.norm(torch.argmax(train_target,dim=1)-torch.argmax(train_prediction_batch,dim=1),ord=0)
      """
      Check the accuracy of the trained model
      """
      test_prediction_batch = NN.forward(test_input) 
      # nb_test_errors = sum(abs(test_target.max(1)[1] - test_prediction_batch.max(1)[1]))
      nb_test_errors = np.linalg.norm(torch.argmax(test_target,dim=1)-torch.argmax(test_prediction_batch,dim=1),ord=0)

    
      print('{:d} loss: {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,loss_output,
                  (100 * nb_train_errors) / nb_train_input,
                  (100 * nb_test_errors) / nb_test_input))

init.plot_results(option=option,
                  nb_train_input=nb_train_input,
                  train_input=train_input, 
                  train_target=train_target, 
                  train_prediction=train_prediction_batch,
                  nb_test_input=nb_test_input,
                  test_input=test_input, 
                  test_target=test_target,
                  test_prediction=test_prediction_batch)
