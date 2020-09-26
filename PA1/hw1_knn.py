from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy
from collections import Counter


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
import heapq
import hw1_utils

class KNN:
    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: Complete the training function
    def train(self, features: List[List[float]], labels: List[int]):
        #raise NotImplementedError
        self.features = features
        self.labels = labels
    
    #TODO: Complete the prediction function
    # Function PREDICT: input is a list of list of float for features, output is a list of int for labels
    def predict(self, features: List[List[float]]) -> List[int]:
        #raise NotImplementedError

        # convert features input to array
        X = np.asarray(features)

        # convert training data input to array
        trainX = np.asarray(self.features)
        trainY = np.asarray(self.labels)

        # Size of dataset
        # number of training data:
        numTrain = trainX.shape[0]

        # number of test/valdation data:
        numTest = X.shape[0]

        # number of attributes of data
        numFeatures = X.shape[1]

        # Validate that same amount of attributes are present in training and validation data
        assert X.shape[1] == trainX.shape[1]

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

        # Initiate label for each test instance using KNN method
        knnLabels = np.zeros((numTest, 1))

        # For each validation test, choose k labels of k smallest distances
        for i in range(numTest):
            # Count in sortedLabels matrix the appearances of each label value based on hyperparameter k of KNN
            countLabels = Counter(sortedLabels[i, :self.k])

            # Decide the knn label based on the most common label
            knnLabels[i, 0] = countLabels.most_common(1)[0][0]
        return knnLabels
        
    #TODO: Complete the get k nearest neighbor function
    def get_k_neighbors(self, point):
        #raise NotImplementedError

        
    #TODO: Complete the model selection function where you need to find the best k     
    def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        
            
            #Dont change any print statement
            print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                      'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    
            print()
            print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
                  'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
            print()
            return best_k, model
    
    #TODO: Complete the model selection function where you need to find the best k with transformation
    def model_selection_with_transformation(distance_funcs,scaling_classes, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        
                #Dont change any print statement
                print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                          'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) + 
                          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    
                print()
                print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
                      'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
                print()
        
        
    #TODO: Do the classification 
    def test_classify(model):
        

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
