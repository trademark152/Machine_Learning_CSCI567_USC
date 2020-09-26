import numpy as np
import matplotlib.pyplot as plt
from typing import List

#TODO: Information Gain function
def Information_Gain(branches):
    # branches: List[List[any]]
    # return: float
    pass

# TODO: implement reduced error pruning
def reduced_error_pruning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List[any]
    pass

# print current tree
# Do not change this function
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    for idx_cls in range(node.num_cls):
        string += str(node.labels.count(idx_cls)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#KNN Utils

#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """

    # check data input quality
    assert len(real_labels) == len(predicted_labels)

    # Convert to array
    real_labels = np.asarray(real_labels)
    predicted_labels = np.asarray(predicted_labels)

    # Calculate TP, FP, FN instances
        # TP is when predicted_labels is the same with real_labels as value 1
        # FP is when predicted = 1 but real = 0
        # FN is when predicted  = 0 but real = 1
    # initiate counts
    tp = 0
    fp = 0
    fn = 0
    # loop through data
    for i in range(len(real_labels)):
        if real_labels[i] == 1 and predicted_labels[i] == 1:
            tp += 1
        elif real_labels[i] == 0 and predicted_labels[i] == 1:
            fp += 1
        elif real_labels[i] == 1 and predicted_labels[i] == 0:
            fn += 1

    # If the data is binary classified, there are other ways to calculate fp and fn
    # Below are methods to cross check the result
    # # False positive = all predicted - true positive
    # fp1 = sum(predicted_labels) - tp
    # assert fp1 == fp
    # # False negative = all positive - true positive
    # fn1 = sum(real_labels) - tp
    # assert fn1 == fn

    # Add one smoothing to prevent zeros in precision and recall calculation
    if tp == 0:
        # Precision:
        p = 1

        # Recall:
        r = 1
    else:
        # Precision: fraction of relevant instances (tp) among the retrieved instances (tp+np)
        p = tp / (tp + fp)

        # Recall: fraction of relevant instances (tp) that have been retrieved over the total amount of relevant instances (tp+fn)
        r = tp / (tp + fn)

    # calculate f1 score: harmonic average of precision and recall
    f1Score = 2 / (1 / p + 1 / r)
    return f1Score
    
    
#TODO: Euclidean distance, inner product distance, gaussian kernel distance and cosine similarity distance

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Euclidian
    # distEuclidean = np.sqrt((x1-x2)*(x1-x2))
    distEuclidean = np.linalg.norm(x1-x2,ord=2)
    return distEuclidean


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Inner Product
    #distInnerProduct = np.dot(x1,x2)
    distInnerProduct = np.inner(x1, x2)
    return distInnerProduct


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Gaussian Kernel d(x,y) = exp[-1/2*sqrt[(x-y)*(x-y)]]
    #distGaussianKernel = np.exp(-1/2*np.sqrt((x1-x2)*(x1-x2)))
    distEuclidean = np.linalg.norm(x1 - x2, ord=2)
    distGaussianKernel = -np.exp(-1 / 2 * distEuclidean)
    return distGaussianKernel


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Cosine similarity:
    distInnerProduct = np.inner(x1, x2)
    distCosineSimilarity = distInnerProduct / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return distCosineSimilarity

# Class to normalize attributes of input
class NormalizationScaler:
    # initiate newly created object
    def __init__(self):
        pass

    # implement function call operator
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        # raise NotImplementedError

        # convert attributes to arrays
        inputArray = np.asarray(features)

        # specify dimensions:
        numRow = inputArray.shape[0]
        numColumn = inputArray.shape[1]

        # initiate matrix of normalized input
        normalizedFeatures = np.zeros((numRow, numColumn))

        for idx, feature in enumerate(inputArray):
            # calculate vector norm
            norm = np.linalg.norm(feature)

            # normalize each instance's feature vector:
            if norm != 0:
                normalizedFeatures[idx, :] = np.divide(feature, norm)

            # if input norm is 0, keep zero values
            else:
                normalizedFeatures[idx, :] = np.zeros((1, numColumn))
        return list(normalizedFeatures.tolist())

class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.
    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]
        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        # raise NotImplementedError

        # initialize min and max values of features
        minMaxFeatures = list()

        # loop through each kind of features
        for feature in range(len(features[0])):
            # loop through each instance and extract the value of a particular feature
            featureVal = [instance[feature] for instance in features]

            # get the min and max value of each column of a particular feature
            featureValMin = min(featureVal)
            featureValMax = max(featureVal)

            # append to the pre-defined list
            minMaxFeatures.append([featureValMin, featureValMax])

        # Min max scaling is scaledX = (X - Xmin) / (Xmax - Xmin)
        # loop through each instance, then loop through each feature to perform scaling
        for instance in features:
            for feature in range(len(instance)):
                instance[feature] = (instance[feature] - minMaxFeatures[feature][0]) / (minMaxFeatures[feature][1] - minMaxFeatures[feature][0])

        return features
