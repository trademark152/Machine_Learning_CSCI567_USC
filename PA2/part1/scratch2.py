from linear_regression import linear_regression_noreg,linear_regression_invertible, regularized_linear_regression, tune_lambda, mean_square_error,mapping_data
from data_loader import data_processing_linear_regression
import numpy as np

filename = 'winequality-white.csv'

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




X=[[1,2,1],[2,1,2],[3,1,1]]
# print(X)
power = 3
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, True, power)
print(mapping_data(Xtrain,1).shape)
print(mapping_data(Xtrain,power).shape)
print(mapping_data(X,1))
print(mapping_data(X,power))

# print(Y)

# Z = []
# for idx, row in enumerate(Y):
#     c = [item for pair in zip(row, X3[idx]) for item in pair]
#     Z.append(c)
# print(Z)