import torch
import numpy as np
from collections import OrderedDict
#from abc import ABC, abstractmethod


# Parent Module class:
# Define the structure of a general Module object
class Module(object):

  def forward(self, input):
    raise NotImplementedError

  def backward(self, gradwrtoutput):
    raise NotImplementedError

  def param(self, eta):
      return []

class Sigmoid(Module):

  def forward(self, input):
    self.input = input[0]
    self.output = 1/(1+np.exp(-self.input))
    return self.output

  def backward(self, gradwrtoutput):
    self.gradwrtinput = 1+np.exp(-self.output)*gradwrtoutput
    return self.gradwrtinput

  def param(self,eta):
      return []

class Tanh(Module):

  def forward(self, input):
    self.input = input[0]
    self.output = np.tanh(self.input)
    return self.output

  def backward(self, gradwrtoutput):
    self.gradwrtinput = (1-np.square(np.tanh(self.output)))*gradwrtoutput
    return self.gradwrtinput

  def param(self,eta):
      return []  

class ReLu(Module):

  def forward(self, input):
    self.input = input[0]
    self.output = np.maximum(self.input,0)
    return self.output

  def backward(self, gradwrtoutput):
    self.gradwrtinput = self.input
    self.gradwrtinput[self.gradwrtinput<=0] = 0
    self.gradwrtinput[self.gradwrtinput>0] = 1
    self.gradwrtinput = self.gradwrtinput*gradwrtoutput
    return self.gradwrtinput

  def param(self,eta):
    return [] 

class Sum(Module):

  def forward(self,input):
    self.input1 = input[0]
    self.input2 = input[1] 
    self.output = self.input1 + self.input2
    return self.output

  def backward(self,gradwrtoutput):
    self.gradwrtinput = 0.5*gradwrtoutput
    return self.gradwrtinput

  def param(self,eta):
    return []


class Linear(Module):

  def __init__(self, *dim):
    self.weights = torch.randn(*dim)
    self.bias = torch.randn(1,*dim[:-1])
    self.gradwrtweights = torch.zeros(*dim)
    self.gradwrtbias = torch.zeros(1,*dim[:-1])

  def forward(self, input):
    self.input = input[0]
    self.output = self.input.mm(self.weights.t())+self.bias
    return self.output

  def backward(self, gradwrtoutput):
    # Remark: gradwrtoutput is dl/ds^(l+1) - should be 10x2
    # Compute gradient with respect to input: dl/dx = w^(l+1)^T * dl/ds^(l+1)
    self.gradwrtinput = torch.mm(gradwrtoutput,self.weights) # weights 2x2 # should be 10x2 (same size as input x)
    # Compute derivatives of loss wrt parameters: dl/dx^(l) = dl/ds^(l)*x^(l-1)^T
    #self.gradwrtweights = torch.mm(self.input.t(),gradwrtoutput)/self.input.size(0)
    #self.gradwrtbias = torch.mean(gradwrtoutput,0) #dl/db^(l) = dl/dx^(l) (in the linear case) we take the mean over the samples

    self.gradwrtweights = torch.mm(gradwrtoutput.t(),self.input)/self.input.size(0)
    #self.gradwrtbias = torch.mean(gradwrtoutput,0) #dl/db^(l) = dl/dx^(l) (in the linear case) we take the mean over the samples
    self.gradwrtbias = torch.sum(gradwrtoutput,0)/self.input.size(0)
    return self.gradwrtinput

  def param(self,eta):
      # Update the parameters
      self.weights = self.weights-eta*self.gradwrtweights
      self.bias = self.bias-eta*self.gradwrtbias

      return []

class LossMSE(Module):

  def forward(self, predicted, true):
    self.diff = predicted - true
    # loss = torch.sum(torch.pow(self.diff,2),1)
    loss = torch.mean(torch.pow(self.diff,2))
    return loss 

  def backward(self):
    self.gradwrtinput = 2*self.diff
    return self.gradwrtinput

  def param(self,eta):
    return []


class Sequential(Module):

  def __init__(self,operators,connectivity,input_operators,output_operator,loss,eta):
    self.operators = operators
    self.connectivity = connectivity
    self.input_operators = input_operators
    self.output_operator = output_operator

    self.connectivity_backward = OrderedDict(reversed(list(self.connectivity.items())))
    
    self.connectivity_forward = {}
    for i in self.operators.keys():
      if i not in self.input_operators:
        tp_forward = ()
        for j in self.connectivity:
          if i in self.connectivity[j]:
            tp_forward = tp_forward + (j,)
        self.connectivity_forward[i] = tp_forward
    self.connectivity_forward = OrderedDict(self.connectivity_forward)
   
    print(connectivity)
    print(self.connectivity_forward)
    print(self.connectivity_backward)
    asdas
    # print(self.connectivity)
    # print(self.connectivity_backward)
    # print(self.connectivity_forward)
    # adsad

    # inv_map = invert_dict(self.connectivity)
    # self.inv_connectivity = OrderedDict(reversed(list(inv_map.items())))


    self.loss = loss
    self.eta = eta

  def forward(self, input):
    """
    
    """
    # for i in self.input_operators:
    #   self.operators[i].forward(input)
    # for opt_i,opt_o_list in self.connectivity.items():
    #   for opt_o in opt_o_list:
    #     self.operators[opt_o].forward(self.operators[opt_i].output)
    # # return self.operators[self.output_operator].output

    for i in self.input_operators:
      self.operators[i].forward([input]) 
    for opt_i,opt_o_list in self.connectivity_forward.items():  
      input_list = []
      for opt_o in opt_o_list:
        input_list.append(self.operators[opt_o].output)
      self.operators[opt_i].forward(input_list)

    return self.operators[self.output_operator].output
  
  def backward(self, gradwrtoutput):
   
    dLdx = self.loss.backward()
    self.operators[self.output_operator].backward(dLdx)
    for opt_i,opt_o_list in self.connectivity_backward.items():
      # print(opt_i)
      # print(opt_o_list)
      # print("\n")
      for opt_o in opt_o_list:
        self.operators[opt_i].backward(self.operators[opt_o].gradwrtinput)
    # input()
    return []

  def param(self):
    for key,value in self.operators.items():
      value.param(self.eta)
    return []


def invert_dict(d):
    result = {}
    for k in d:
        if d[k] not in result:
            result[d[k]] = set()
        result[d[k]].add(k)
    return {k: result[k] if len(result[k])>1 else result[k].pop() for k in result}
