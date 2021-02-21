"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    y_prime=X.dot(w)
    err_arr=np.absolute(y_prime - y)
    N=y.shape[0]
    err = np.sum(err_arr)/N
#    print(type(err))
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  XT=X.transpose()
  M1=XT.dot(X)
  M1=np.linalg.inv(M1)
  w = np.linalg.multi_dot([M1, XT, y])
  return w

def is_invertible(M):
    eig_vals=np.linalg.eigvals(M)
    smallest_eg=np.amin(np.absolute(eig_vals))
    if(smallest_eg < 10**-5):
        return False
    else:
        return True

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    XT=np.transpose(X)
    M1=XT.dot(X)
    while(not is_invertible(M1)):
        M1=M1+((0.1)*np.identity(X.shape[1]))
        
    M1=np.linalg.inv(M1)
    w = np.linalg.multi_dot([M1, XT, y])
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    XT=np.transpose(X)
    M1=XT.dot(X)+(lambd*np.identity(X.shape[1]))
    M1=np.linalg.inv(M1)
    w = np.linalg.multi_dot([M1, XT, y])
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    global_min_err=float('inf')
    global_min_lambda=None
    for i in range(-19,20,1):
        wx=regularized_linear_regression(Xtrain, ytrain,10**i)
        err=mean_absolute_error(wx,Xval, yval)
#        print(err," ",10**i," ", type(err))
        if(err<=global_min_err):
            global_min_err=err
            global_min_lambda=10**i
            
    bestlambda = global_min_lambda
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    newX=np.copy(X)
    if(power>=2):
        for i in range(2,power+1):
            powerX=np.power(X,i)
            newX=np.concatenate((newX,powerX),axis=1)
#    print(X.shape,newX.shape)
    
    return newX


