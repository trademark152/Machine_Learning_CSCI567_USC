from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
from collections import Counter

class KNN:

    def __init__(self, k: int, distance_function):
        # initialize hyper parameter k,
        self.k = k

        # kind of distance function
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        # import features and labels of training instances
        self.features = features
        self.labels = labels

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        # initialize list of predicted labels
        predictedLabels = []

        # loop through each instance:
        for instance in features:
            # get k nearest points' labels of that instance
            kNearestLabels = self.get_k_neighbors(instance)
            # print(kNearestLabels)

            # use counter to count the occurrences of each label in kNN
            countLabels = Counter(kNearestLabels)
            # print(countLabels)

            # determine the kNN label and corresponding count by majority vote
            label, count = countLabels.most_common()[0]

            # get the predicted label by majority vote
            # predictedLabels.append(self.get_label_from_kNN(kNearestLabels))
            predictedLabels.append(label)
        return predictedLabels

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        # convert features input to array
        X = np.asarray([point]) # for single point
        # print(X)

        # convert training data input to array
        trainX = np.asarray(self.features)
        trainY = np.asarray(self.labels)

        # Size of dataset
        # number of training data:
        numTrain = trainX.shape[0]

        # number of test/valdation data:
        numTest = 1  # for single point

        # number of attributes of data
        # numFeatures = X.shape[1]  # for multiple points
        numFeatures = len(point)  # for single point
        # print(numFeatures)

        # Validate that same amount of attributes are present in training and validation data
        # assert X.shape[1] == trainX.shape[1] # for multiple point
        assert numFeatures == trainX.shape[1]

        # Initiate distance matrix between training data and test data
        distances = np.zeros((numTest, numTrain))

        # Loop through dataset to calculate distance btw each training and data
        for i in range(numTest):
            for j in range(numTrain):
                distances[i, j] = self.distance_function(X[i, :], trainX[j, :])

        # Create corresponding label matrix of training instances with dimensions numTest*trainY
        labels = np.tile(trainY, (numTest, 1))

        # Initiate sorted label matrix for each test instance (row)
        sortedLabels = np.zeros((numTest, numTrain))

        # Sort each row of distance matrix: "idx" is row index, "row" is actual value of each row
        for idx, row in enumerate(distances):
            # Extract corresponding labels of that row
            rowLabels = labels[idx, :]

            # Sort each row's value of distances in increasing order and extract corresponding index
            indexIncrease = np.argsort(row)

            # Extract corresponding labels:
            sortedLabels[idx, :] = rowLabels[indexIncrease]

        kNearestLabels = sortedLabels[:, :self.k]
        # print(kNearestLabels)
        return(kNearestLabels[0].tolist())  # because of only single point in get_k_neighbors


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
