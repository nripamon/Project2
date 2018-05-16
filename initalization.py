import torch
import numpy as np
from torchvision import datasets

import matplotlib.pyplot as plt

def initialize_dataset(option):
    number_of_classes = 2 
    

    if option == 0: 

        number_of_inputs = 2
        number_of_classes = 2
        nb_train_input = 1000
        nb_test_input = 200
        # We identify the target line by means of its slope and intercept.
        intercept = 0
        slope = 4
        train_input = torch.randn(nb_train_input,number_of_inputs)
        train_target = torch.zeros(nb_train_input,number_of_classes)
        max_out = 0.8
        min_out = 0.2
        # Iterate over train sample.
        for i in range(nb_train_input):
            if  train_input[i,1]>slope*train_input[i,0]+intercept: 
                train_target[i,0] = min_out
                train_target[i,1] = max_out
            else: 
                train_target[i,0] = max_out
                train_target[i,1] = min_out

        test_input = torch.randn(nb_test_input,number_of_inputs)
        test_target = torch.zeros(nb_test_input,number_of_classes)
        # Iterate over the test sample.
        for i in range(nb_test_input):
            if  test_input[i,1]>slope*test_input[i,0]+intercept: 
                test_target[i,0] = min_out
                test_target[i,1] = max_out
            else: 
                test_target[i,0] = max_out
                test_target[i,1] = min_out
    
    elif option==1:

        number_of_inputs = 2
        number_of_classes = 2
        nb_train_input = 1000
        nb_test_input = 200
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



    elif option==2:
        number_of_inputs = 3
        number_of_classes = 2
        nb_train_input = 1000
        nb_test_input = 200
        """
        Initialize train and test sets
        Given that the loss function reaches 1 and 0 asyntotically, 
        we use 0.8 and 0.2.
        """
        center_x = 0
        center_y = 0
        radius = 1/np.sqrt(2*np.pi)
        
        train_input = torch.rand(nb_train_input,2)
        prod_tr=train_input*train_input
        prod_output = prod_tr[:,0]+prod_tr[:,1]
        train_input = torch.cat((train_input,prod_output.unsqueeze(1)),1)
        train_target = torch.zeros(nb_train_input,number_of_classes)
        for i in range(nb_train_input):
            if ((train_input[i,0]-center_x)**2+(train_input[i,1]-center_y)**2>radius**2): 
                train_target[i,0] = 0
                train_target[i,1] = 1
            else: 
                train_target[i,0] = 1
                train_target[i,1] = 0

        test_input = torch.rand(nb_test_input,2)
        prod_test=test_input*test_input
        prod_output = prod_test[:,0]+prod_test[:,1]
        test_input = torch.cat((test_input,prod_output.unsqueeze(1)),1)
        test_target = torch.zeros(nb_test_input,number_of_classes)
        for i in range(nb_test_input):
            if ((test_input[i,0]-center_x)**2+(test_input[i,1]-center_y)**2>radius**2):  
                test_target[i,0] = 0
                test_target[i,1] = 1
            else: 
                test_target[i,0] = 1
                test_target[i,1] = 0

    elif option==3:
        mnist_train_set = datasets.MNIST('MNIST/', train = True, download = True)
        mnist_test_set = datasets.MNIST('MNIST/', train = False, download = True)

        number_of_inputs = 28*28
        number_of_classes = 10
        nb_train_input = 60000
        nb_test_input = 10000
        # train_input = mnist_train_set.train_data.view(-1, 1, 28, 28).float()
        train_input = mnist_train_set.train_data.view(nb_train_input, 28*28).float()
        train_target_regr = mnist_train_set.train_labels
        train_target = torch.zeros(nb_train_input,10)
        for i in range(nb_train_input):
        	train_target[i,train_target_regr[i]] = 1
        test_input = mnist_test_set.test_data.view(nb_test_input, 28*28).float()
        test_target_regr = mnist_test_set.test_labels
        test_target = torch.zeros(nb_test_input,10)
        for i in range(nb_test_input):
        	test_target[i,test_target_regr[i]] = 1


    return train_input, train_target, test_input, test_target, number_of_inputs, number_of_classes, nb_train_input, nb_test_input


def plot_results(option,nb_train_input,train_input, train_target,train_prediction, nb_test_input,test_input, test_target, test_prediction):

    if option == 0:
        intercept = 0
        slope = 4
        """
        Plot the data for training and test sets
        """
        x_plot = np.linspace(-5,5,10)
        y_plot = slope*x_plot + intercept
        # TRAIN
        min_x_train = torch.min(train_input[:,0])
        max_x_train = torch.max(train_input[:,0])
        min_y_train = torch.min(train_input[:,1])
        max_y_train = torch.max(train_input[:,1])
        train_res = abs(train_target.max(1)[1] - train_prediction.max(1)[1])
        plt.figure(1)
        for i in range(nb_train_input):
          if (train_prediction[i,0] >= train_prediction[i,1]):
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
        plt.plot(x_plot,y_plot,color='black')
        plt.xlim(min_x_train,  max_x_train)
        plt.ylim(min_y_train,  max_y_train)
        plt.show()

        # TEST
        min_x_test = torch.min(test_input[:,0])
        max_x_test = torch.max(test_input[:,0])
        min_y_test = torch.min(test_input[:,1])
        max_y_test = torch.max(test_input[:,1])
        test_res = abs(test_target.max(1)[1] - test_prediction.max(1)[1])
        plt.figure(2)
        for i in range(nb_test_input):
          if (test_prediction[i,0] >= test_prediction[i,1]):
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
        plt.plot(x_plot,y_plot,color='black')
        plt.xlim(min_x_test,  max_x_test)
        plt.ylim(min_y_test,  max_y_test)
        plt.show()


    elif option>=1 :

        center_x = 0
        center_y = 0
        radius = 1/np.sqrt(2*np.pi)
        """
        Plot the data for training and test sets
        """
        # TRAIN
        min_x_train = torch.min(train_input[:,0])
        max_x_train = torch.max(train_input[:,0])
        min_y_train = torch.min(train_input[:,1])
        max_y_train = torch.max(train_input[:,1])
        
        train_res = abs(train_target.max(1)[1] - train_prediction.max(1)[1])
        plt.figure(1)
        for i in range(nb_train_input):
          if (train_prediction[i,0] >= train_prediction[i,1]):
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
        test_res = abs(test_target.max(1)[1] - test_prediction.max(1)[1])
        plt.figure(2)
        for i in range(nb_test_input):
          if (test_prediction[i,0] >= test_prediction[i,1]):
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