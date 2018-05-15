import Module_base as M
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

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
nb_train_input = 1000
nb_test_input = 200

"""
Create List of Modules.
This list is then given as input to the constructor of the NN.
A loss function has to be provided separately.
"""
number_of_inputs = 2
hidden_layer_1 = 25
hidden_layer_2 = 25
hidden_layer_3 = 25
number_of_classes = 2
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
"""
Define the loss function and initialize the neural network.
"""
loss = M.LossMSE()
NN = M.Sequential(operators,connectivity,input_operators,output_operator,loss,eta=0.02)

"""
Initialize train and test sets
Given that the loss function reaches 1 and 0 asyntotically, 
we use 0.8 and 0.2.
"""
center_x = 0
center_y = 0
radius = 1/np.sqrt(2*np.pi)
train_input = torch.rand(nb_train_input,number_of_inputs)
train_target = torch.zeros(nb_train_input,number_of_classes)
for i in range(nb_train_input):
    if ((train_input[i,0]-center_x)**2+(train_input[i,1]-center_y)**2>radius**2): 
        train_target[i,0] = 0
        train_target[i,1] = 1
    else: 
        train_target[i,0] = 1
        train_target[i,1] = 0

test_input = torch.rand(nb_test_input,number_of_inputs)
test_target = torch.zeros(nb_test_input,number_of_classes)
for i in range(nb_test_input):
    if ((test_input[i,0]-center_x)**2+(test_input[i,1]-center_y)**2>radius**2):  
        test_target[i,0] = 0
        test_target[i,1] = 1
    else: 
        test_target[i,0] = 1
        test_target[i,1] = 0

"""
Set the number of epochs
"""
nb_epochs = 4000

for k in range(0, nb_epochs):
    """
    During each epoch we have the following steps:
    - do the forward pass using the training set
    - compute the related loss function
    - do the backward pass to compute the gradients 
      wrt the weights and biases
    - update the weights and biases
    """
    train_prediction = NN.forward(train_input) 
    loss_output = NN.loss.forward(train_prediction,train_target)
    NN.backward(train_prediction)
    NN.param()
    """
    After the training, we compute the number of 
    samples, in the training, that have been correctly
    predicted
    """
    nb_train_errors = sum(abs(train_target.max(1)[1] - train_prediction.max(1)[1]))
    """
    Check on the test the accuracy of the trained model
    """
    test_prediction = NN.forward(test_input) 
    nb_test_errors = sum(abs(test_target.max(1)[1] - test_prediction.max(1)[1]))
    """
    Print on terminal the loss function and the error percentages
    for the prediction for train and test sets
    """
    print('{:d} loss: {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  loss_output,
                  (100 * nb_train_errors) / nb_train_input,
                  (100 * nb_test_errors) / nb_test_input))

"""
Plot the data for training and test sets
"""
# TRAIN
min_x_train = torch.min(train_input[:,0])
max_x_train = torch.max(train_input[:,0])
min_y_train = torch.min(train_input[:,1])
max_y_train = torch.max(train_input[:,1])
train_pred = NN.forward(train_input)
train_res = abs(train_target.max(1)[1] - train_pred.max(1)[1])
plt.figure(1)
for i in range(nb_train_input):
  if (train_pred[i,0] >= train_pred[i,1]):
    c = 'red'
  else:
    c = 'blue'
  s = 2
  m = '.'
  if (train_res[i] == 1):
    s = 10
    m = 'x'
    c = 'black'
  plt.scatter(train_input[i,0],train_input[i,1],color=c,marker=m,s=s)
plt.Circle((center_x,center_y),radius,color='black')
plt.xlim(min_x_train,  max_x_train)
plt.ylim(min_y_train,  max_y_train)
plt.show()

# TEST
min_x_test = torch.min(test_input[:,0])
max_x_test = torch.max(test_input[:,0])
min_y_test = torch.min(test_input[:,1])
max_y_test = torch.max(test_input[:,1])
test_pred = NN.forward(test_input)
test_res = abs(test_target.max(1)[1] - test_pred.max(1)[1])
plt.figure(2)
for i in range(nb_test_input):
  if (test_pred[i,0] >= test_pred[i,1]):
    c = 'red'
  else:
    c = 'blue'
  s = 2
  m = '.'
  if (test_res[i] == 1):
    s = 10
    m = 'x'
    c = 'black'
  plt.scatter(test_input[i,0],test_input[i,1],color=c,marker=m,s=s)
plt.Circle((center_x,center_y),radius,color='black')
plt.xlim(min_x_test,  max_x_test)
plt.ylim(min_y_test,  max_y_test)
plt.show()