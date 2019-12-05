"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    # Calculate mean square error
    # MSE = 1/n * sum [(y_true-y_pred)^2]
    # Dimension: X: num_samples*D; y: num_samples
    err = np.mean(np.power(np.subtract(y, np.matmul(X,w)),2))
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
  # Closed form solution: w=(Xt*X)^-1*Xt*y
  # Covariance matrix
  covMat = np.matmul(np.transpose(X), X)

  # weight vector
  w = np.matmul(np.matmul(np.linalg.inv(covMat), np.transpose(X)),y)
  return w

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
    # Number of dimensions
    dim = len(X[0])
    # print(dim)

    # Covariance matrix
    covMat = np.matmul(np.transpose(X), X)

    # Find eigenvalues:
    eigVals = np.linalg.eigvals(covMat)
    # print(eigVals)
    # print(np.amin(np.absolute(eigVals)))

    if np.amin(np.absolute(eigVals)) >= 10**(-5):
        # weight vector
        return np.matmul(np.matmul(np.linalg.inv(covMat), np.transpose(X)), y)

    # If the smallest absolute value of any eigenvalue is smaller than 10^-5
    # Consider matrix non-invertibale and start improving:
    k = 0
    while np.amin(np.absolute(eigVals)) < 10**(-5):
        # solve issue of non-invertible (slides 29-31 csci567 lecture 3)
        k += 1
        eigVals = np.linalg.eigvals(covMat+k*10**(-1)*np.identity(dim))

    # print(k)
    return np.matmul(np.matmul(np.linalg.inv(covMat+k*(10**(-1))*np.identity(dim)), np.transpose(X)), y)



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
    # handle exception
    # if lambd == None:
    #     lambd = 0.

    # Number of dimensions
    dim = len(X[0])
    # print(dim)

    # Covariance matrix
    covMat = np.matmul(np.transpose(X), X)

    # # Find eigenvalues:
    # eigVals = np.linalg.eigvals(covMat)
    # # print(eigVals)
    # # print(np.amin(np.absolute(eigVals)))

    # # if matrix is invertible
    # if np.amin(np.absolute(eigVals)) >= 10**(-5):
    #     # weight vector
    #     return np.matmul(np.matmul(np.linalg.inv(covMat), np.transpose(X)), y)
    #
    # # If the smallest absolute value of any eigenvalue is smaller than 10^-5
    # # Consider matrix non-invertibale and start improving:
    # else:
    #     # solve issue of non-invertible (slides 50 csci567 lecture 3)
    #     eigVals = np.linalg.eigvals(covMat+lambd*np.identity(dim))

    return np.matmul(np.matmul(np.linalg.inv(covMat+lambd*np.identity(dim)), np.transpose(X)), y)


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
    bestlambda = -1
    lowestMSE = np.inf
    lambd = 10**(-20)

    while lambd < 10**20:
        # update lambd
        lambd *= 10
        # print(float("{0:.2e}".format(lambd)))

        # use given training data to train model
        w = regularized_linear_regression(Xtrain, ytrain, lambd)

        # compute the mse
        mse = mean_square_error(w, Xval, yval)
        # print(mse)

        # update the mse
        if mse < lowestMSE:
            lowestMSE = mse
            bestlambda = lambd

    if bestlambda == None:
        return 0
    else:
    # print(bestlambda)
        # avoid representation error in floating number
        return float("{0:.2e}".format(bestlambda))
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, shape(num_samples, D*power) You can manually calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    """ GOAL: input [[1,2,3],[0,5,5]] --> output [[1,2,3,1,4,9],[0,5,5,0,25,25]]"""
    # loop through each training sample
    # mapped_X = np.zeros((len(X), len(X[0])*(power-1)))
    mapped_X = [[] for i in range(len(X))]
    # mapped_X=[]
    # print(mapped_X)
    for index, sample in enumerate(X):
        # print(sample)

        # loop through all power in range
        for i in range(2, power+1):
            # create an element-wise power of the original sample
            sample_power_i = np.power(sample[:len(X[0])], i)
            # print(sample_power_i)

            # obtain the index of the last element
            end_idx = len(sample)
            # print(end_idx)

            # add that to the end of the original row
            sample = np.insert(sample, end_idx, sample_power_i)
        # print(sample.tolist())

        # modify X
        mapped_X[index] = sample
    return np.asarray(mapped_X)


