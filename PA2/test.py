import numpy as np
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
    for index, row in enumerate(X):
        # print(row)
        for i in range(2, power):
            rowi = np.power(row, i)
            # print(rowi)
            idx = len(row)
            # print(idx)
            row = np.insert(row, idx, rowi)
        # print(row)
        X[index] = row.tolist()
    return X


X=[[1,2,3],[0,5,5]]
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, False, True, i)
power = 20
print(mapping_data(X,power))


# print(Y)

# Z = []
# for idx, row in enumerate(Y):
#     c = [item for pair in zip(row, X3[idx]) for item in pair]
#     Z.append(c)
# print(Z)