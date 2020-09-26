import numpy as np
from collections import Counter
from hw1_knn import KNN
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance
from utils import f1_score
import utils

distance_funcs = {
    'euclidean': euclidean_distance,
    'gaussian': gaussian_kernel_distance,
    'inner_prod': inner_product_distance,
    'cosine_dist': cosine_sim_distance,
}

# Test f1_score
realLabels =      [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
predictedLabels = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
print(f1_score(realLabels, predictedLabels))

realLabels = [1,1,1,0]
predictedLabels = [1,0,0,1]
print(f1_score(realLabels, predictedLabels))
# Test distance
# x1=[1,2,3]
# x2=[-2,1,4]
# x3=[3,2,5]
# x4=[1,-2,3]
# x5=[-2,4,1]

x1=[1,1]
x2=[2,3]
x3=[3,2]
x4=[4,2]
x5=[5,3]

xTrain =[x1,x2,x3,x4,x5]
yTrain = [0,0,1,1,1]

# Xtrain = [[1, 1], [2, 3], [3, 2], [4, 2], [5, 3]]
# ytrain = [0, 0, 1, 1, 1]
# Xval = [[1, 2], [5, 2]]
# yval = [0, 1]
# Xtest = [[1, 3], [5, 1]]
# ytest = [0, 1]

x6=[[3, 4], [1, -1], [0, 0]]
x7=[[2, -1], [-1, 5], [0, 0]]

print(euclidean_distance(x1,x2))
print(inner_product_distance(x1,x2))
print(gaussian_kernel_distance(x1,x2))
print(cosine_sim_distance(x1,x2))
#
# # test normalize of AN input
Normalize = utils.NormalizationScaler()
print(x6)
x6Norm = Normalize(x6)
# print(Normalize(x12))
print(x6)
print(x6Norm)


#
MinMax = utils.MinMaxScaler()
print(x7)
# print(MinMax(x21))
x7MinMax = MinMax(x7)
# print(Normalize(x12))
print(x7)
print(x7MinMax)

train_features = [[0, 10], [2, 0]]
test_features = [[20, 1]]

scaler = utils.MinMaxScaler()
train_features_scaled = scaler(train_features)
print(train_features_scaled)
# now train_features_scaled should be [[0, 1], [1, 0]]

test_features_scaled = scaler(test_features)
print(test_features_scaled)
# now test_features_scaled should be [[10, 0.1]]

new_scaler = utils.MinMaxScaler()  # creating a new scaler
train_features_scaled2 = new_scaler([[1, 1], [0, 0]])  # new trainfeatures
print(train_features_scaled2)
test_features_scaled2 = new_scaler(test_features)
print(test_features_scaled2)
# now test_features_scaled should be [[20, 1]]

#
# # Test KNN:
import numpy as np
from hw1_knn import KNN
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance
from utils import f1_score, model_selection_without_normalization, model_selection_with_transformation


from data import data_processing
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
# print(Xtrain)
# print(ytrain)
# print(Xval)
# print(yval)

# Xtrain1 = [[1,1],[1,4],[2,1],[2,2],[2,4],[3,1],[3,3],[4,2],[5,1],[5,4]]
# ytrain1 = [0,1,0,1,0,1,1,1,0,0]
# kNN = KNN(1,distance_funcs['euclidean'])
# kNN.train(Xtrain1, ytrain1)
#
# print(Xtrain1)
# print(ytrain1)
# print(kNN.get_k_neighbors([1,2]))
# print(kNN.predict([[1,2],[3,2]]))

best_model, best_k, best_function = model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval)
# print(best_k)
# print(best_model)
# print(best_function)
# Xtrain, ytrain

from utils import NormalizationScaler, MinMaxScaler

scaling_classes = {
    'min_max_scale': MinMaxScaler,
    'normalize': NormalizationScaler,
}

best_model, best_k, best_function, best_scaler = model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval)
# print(best_k)
# print(best_model)
# print(best_function)




# from data import data_processing
# Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()

# print(Xtrain)
# print(ytrain)
# print(Xval)
# print(yval)
# print(Xtest)
# print(ytest)

# numTest = 3
# labels = np.tile(ytrain, (numTest, 1))
# print(labels)
# for idx, row in (enumerate(labels)):
#     print(idx)
#     print(row)
#     print(np.argsort(row))
#     row1 = row[np.argsort(row)]
#     print(row1)
#     print(Counter(row1[:2]))

