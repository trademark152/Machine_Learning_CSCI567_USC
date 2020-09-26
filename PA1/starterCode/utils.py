import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float: indicating entropy of set S
    # branches: List[List[int]] num_branches * num_classes
    # return: float
    # raise NotImplementedError

    '''
    branches: num_branches array * num_classes,
    		  C is the number of classes,
    		  B is the number of branches
    		  it stores the number of
    		  corresponding training samples
    		  e.g.
    		              ○ ○ ○ ○
    		              ● ● ● ●
    		            ┏━━━━┻━━━━┓
    	               ○ ○       ○ ○
    	               ● ● ● ●

    	      branches = [[2,4], [2,0]]
    '''
    # calculate the number of ALL examples
    print(S)
    total_examples = 0
    for branch in branches:
        total_examples += np.sum(branch)

    # return 0 if there is no example
    if total_examples == 0:
        return 0.0

    # initiate remainder (difference between entropy and IG
    remainder = 0.0

    # loop through each branch
    for branch in branches:
        # calculate probability of that branch
        total_probability = np.sum(branch) / total_examples

        # calculate entropy of that branch
        entropy = get_entropy(branch)

        # calculate remainder term
        remainder += total_probability * entropy
        # print(remainder)

    return S - remainder

# Function to get entropy of  a branch
def get_entropy(branch):
    # get the total_examples example of that branch
    total_examples_branch = np.sum(branch)

    # if there is no example:
    if total_examples_branch == 0:
        return 0

    # initiate answer
    entropy = 0

    # loop through each class in a branch
    for classCount in branch:
        if classCount == 0:
            entropy += 0
        else:
            probability = classCount / total_examples_branch
            entropy += -1*(probability * np.log2(probability))
    return entropy

# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    # raise NotImplementedError
    pass
    # def prune(self, tree, X_test):
    # if have no test data collapse the tree
    # if X_test.shape[0] == 0:
    #     return '>50K'
    #
    # left_set = []
    # right_set = []
    # # if the branches are not decisionTrees try to prune them
    # if (is_decisionTree(decisionTree['Right']) or is_decisionTree(decisionTree['Left'])):
    #     left_set, right_set = test_split(decisionTree.columns.index(decisionTree['Node']), decisionTree['Value'], X_test)
    #
    # if is_decisionTree(decisionTree['Left']):
    #     decisionTree['Left'] = prune(decisionTree['Left'], left_set)
    #
    # if is_decisionTree(decisionTree['Right']):
    #     decisionTree['Right'] = prune(decisionTree['Right'], right_set)
    #
    # # if they are now both leafs, see if can merge them
    # if not is_decisionTree(decisionTree['Left']) and not is_decisionTree(decisionTree['Right']):
    #     left_set, right_set = test_split(decisiondecisionTree.columns.index(decisionTree['Node']), decisionTree['Value'], X_test)
    #
    #     if left_set.shape[0] == 0:
    #         left_error_sum = 0
    #     else:
    #         left_error_sum = testing_major(decisionTree['Left'], left_set[:, -1])
    #
    #     if right_set.shape[0] == 0:
    #         right_error_sum = 0
    #     else:
    #         right_error_sum = testing_major(decisionTree['Right'], right_set[:, -1])
    #
    #     error_no_merge = pow(left_error_sum, 2) + pow(right_error_sum, 2)
    #     decisionTree_mean = to_terminal(X_test)
    #     error_merge = pow(testing_major(decisionTree_mean, X_test[:, -1]), 2)
    #
    #     if error_merge < error_no_merge:
    #         # print "merging"
    #         return decisionTree_mean
    #     else:
    #         return decisionTree
    # else:
    #     return decisionTree

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    sumNumerator = 0.0
    sumRealY = 0.0
    sumPredictedY = 0.0

    for index in range(0, len(real_labels)):
        sumNumerator += real_labels[index] * predicted_labels[index]
        sumRealY += real_labels[index]
        sumPredictedY += predicted_labels[index]

    # in case denominator is 0
    if sumRealY + sumPredictedY == 0:
        return 0

    return (2 * sumNumerator) / (sumRealY + sumPredictedY)

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Euclidian
    # distEuclidean = np.sqrt((x1-x2)*(x1-x2))
    distEuclidean = np.linalg.norm(x1-x2, ord=2)
    return distEuclidean


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Inner Product
    # distInnerProduct = np.dot(x1,x2)
    distInnerProduct = float(np.inner(x1, x2))
    return distInnerProduct


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Gaussian Kernel d(x,y) = exp[-1/2*sqrt[(x-y)*(x-y)]]
    #distGaussianKernel = np.exp(-1/2*np.sqrt((x1-x2)*(x1-x2)))
    distEuclidean = np.linalg.norm(x1 - x2, ord=2)
    distGaussianKernel = -np.exp(-1 / 2 * distEuclidean ** 2)
    return distGaussianKernel


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    # convert coordinates of each point to array
    x1 = np.asarray(point1)
    x2 = np.asarray(point2)

    # Distance Cosine similarity:
    distInnerProduct = np.inner(point1, point2)
    distCosineSimilarity = distInnerProduct / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return float(1-distCosineSimilarity)


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # raise NotImplementedError

    # initiate model, best k, current k and best f1 score:
    best_model = None
    best_k = None
    best_f1_score = None
    current_k = 1
    best_func = None
    best_method = None

    # order of preferrence of directions
    order = ["euclidean", "gaussian", "inner_prod", "cosine_dist"]
    #
    #     # return true if index of direction 1 is smaller than index of direction 2 in the above list
    #     return directions.index(distanceFunc1) < directions.index(distanceFunc2)

    # loop until k reaches number of sample -1

    if len(ytrain) < 30:
        max_k = len(ytrain) - 1
    else:
        max_k = 30

    # while current_k < len(ytrain):
    while current_k < max_k:
        # loop through each distance function method:
        for method in distance_funcs.keys():
            distance_func = distance_funcs[method]

            # create the model based on this current k and distance function
            kNNClassifier = KNN(current_k, distance_func)

            # Train this model with training data
            kNNClassifier.train(Xtrain, ytrain)

            # Get f1 score on validation dataset to optimize (best method is the one with highest validation f1 score)
            kNNF1Score = f1_score(yval,kNNClassifier.predict(Xval))

            # Dont change any print statement
            print()
            print('[part 1.1] {name}\tk: {k:d}\t'.format(name=distance_func, k=current_k) +
                   'valid: {valid_f1_score:.5f}'.format(valid_f1_score=kNNF1Score))

            # update best values
            if best_f1_score == None or best_f1_score < kNNF1Score:
                best_f1_score = kNNF1Score
                best_k = current_k
                best_model = kNNClassifier
                best_func = distance_func
                best_method = method

            # break ties by order of preference
            if best_f1_score == kNNF1Score:
                if order.index(method) < order.index(best_method):
                    best_func = distance_func
                    best_f1_score = kNNF1Score
                    best_k = current_k
                    best_model = kNNClassifier

        # start with k = 1 and incrementally increase by 2
        current_k += 2

    print("Best distance method: ", str(best_model.distance_function), " and best k is ", best_k)
    print("Corresponding valid_f1_score: ", str(f1_score(yval,best_model.predict(Xval))))
    print("predicted_yval: ", best_model.predict(Xval))
    print("true_yval: ", yval)
    return best_model, best_k, best_method


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    # raise NotImplementedError

    # initiate model, best k, current k and best f1 score:
    best_model = None
    best_f1_score = None
    best_k = None
    current_k = 1
    best_func = None
    best_scaler = None
    best_method = None
    best_scaling_method = None
    order1 = ['euclidean', 'gaussian', 'inner_prod', 'cosine_dist']
    order2 = ['min_max_scale', 'normalize']

    if len(ytrain) < 30:
        max_k = len(ytrain) - 1
    else:
        max_k = 30

    # loop through different scaling methods
    for scaling_method in scaling_classes.keys():
        scaling_class = scaling_classes[scaling_method]
        print()
        print("Current scaling method: ", scaling_class)

        # create the scaler
        scaler = scaling_class()
        # print(scaler)

        # scale dataset, no need to scale ytrain because values already from 0 to 1
        scaled_Xtrain = scaler(Xtrain)
        scaled_Xval = scaler(Xval)

        # loop until k reaches number of sample -1
        # while current_k < len(ytrain):
        while current_k < max_k:
            # loop through each distance function method:
            for method in distance_funcs.keys():
                distance_func = distance_funcs[method]

                # create the model based on this current k and distance function
                kNNClassifier = KNN(current_k, distance_func)

                # Train this model with training data
                kNNClassifier.train(scaled_Xtrain, ytrain)

                # Get f1 score on validation dataset to optimize (best method is the one with highest validation f1 score)
                kNNF1Score = f1_score(yval,kNNClassifier.predict(scaled_Xval))

                # Dont change any print statement
                print()
                print('[part 1.2] {scaling_name}\t{distance_name}\tk: {k:d}\t'.format(distance_name=distance_func, scaling_name=scaling_class, k=current_k) +
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score=kNNF1Score))

                # update best values
                if best_f1_score == None or best_f1_score < kNNF1Score:
                    best_f1_score = kNNF1Score
                    best_k = current_k
                    best_model = kNNClassifier
                    best_scaler = scaling_class
                    best_func = distance_func
                    best_scaling_method = scaling_method
                    best_method = method

                # break ties
                if best_f1_score == kNNF1Score:
                    if order2.index(scaling_method) < order2.index(best_scaling_method):
                        best_func = distance_func
                        best_f1_score = kNNF1Score
                        best_k = current_k
                        best_model = kNNClassifier
                        best_scaler = scaling_class
                    elif order2.index(scaling_method) < order2.index(best_scaling_method):
                        if order1.index(method) < order1.index(best_method):
                            best_func = distance_func
                            best_f1_score = kNNF1Score
                            best_k = current_k
                            best_model = kNNClassifier
                            best_scaler = scaling_class


            # start with k = 1 and incrementally increase by 2
            current_k += 2

        # reset counter k
        current_k = 1

    print("Best scaling method: ", str(best_scaler), " and best k is ", best_k)
    print("Best distance method: ", str(best_model.distance_function), " and best k is ", best_k)
    print("Corresponding valid_f1_score: ", best_f1_score)
    print("predicted_yval: ", best_model.predict(Xval))
    print("true_yval: ", yval)
    return best_model, best_k, best_method, best_scaling_method



class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
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

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
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
        # initialize states
        self.maxList = []
        self.minList = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """

        # CREATE 2 lists: MAX and MIN VALUES of EACH FEATURE[INDEX}
        # Case where max and min lists are empty, not yet initialized
        if len(self.maxList) == 0 and len(self.minList) == 0:
            for feature in features:
                for idx in range(0, len(feature)):
                    if len(self.maxList) == 0 and len(self.minList) == 0:
                        # if nothing is in these lists, just copy over the feature's values to initialize a list
                        self.maxList = feature.copy()
                        self.minList = feature.copy()
                        break
                    # import new values and compare to the old ones
                    else:
                        self.maxList[idx] = max(self.maxList[idx], feature[idx])
                        self.minList[idx] = min(self.minList[idx], feature[idx])

        # initialize results
        scaledFeatures = []
        # loop through each value to min-max scaling
        for feature in features:
            scaledFeature = []
            for idx in range(0, len(feature)):
                # obtain the min-max value to scale from stored lists
                maxVal = self.maxList[idx]
                minVal = self.minList[idx]
                diff = maxVal - minVal
                
                # if minVal = maxVal # return scaled value as zeros
                if diff == 0:
                    scaledVal = 0

                else:
                    scaledVal = (feature[idx] - minVal) / (maxVal - minVal)

                # update each feature
                scaledFeature.append(scaledVal)
            # update output
            scaledFeatures.append(scaledFeature)
        return scaledFeatures





