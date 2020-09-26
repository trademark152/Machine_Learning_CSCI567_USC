import data
import hw1_dt as decision_tree
import utils as Utils
from sklearn.metrics import accuracy_score
import numpy as np

#TEST IG:
root = [8,12]
branches = [[5,2],[3,10]]
igRoot = Utils.get_entropy(root)
print("IG root",igRoot)
print("IG branches", Utils.Information_Gain(igRoot,branches))

features, labels = data.sample_decision_tree_data()
print(features)
print(labels)

#
# data
X_test, y_test = data.sample_decision_tree_test()
print(X_test)
print(y_test)

# build the tree
dTree = decision_tree.DecisionTree()
dTree.train(features, labels)
# print
Utils.print_tree(dTree)

# testing
y_est_test = dTree.predict(X_test)
print(y_est_test)
test_accu = accuracy_score(y_est_test, y_test)
print('test_accu', test_accu)

# X_train = [alt, bar, fri, hun, pat, price, rain, res, type, est]
# X_train = [[1,0,0,1,'s','$$$',0,1,'f','0-10'],
#            [1,0,0,1,'f','$',0,0,'t','30-60'],
#            [0,1,0,0,'s','$',0,0,'b','0-10'],
#            [1,0,1,1,'f','$',0,0,'t','10-30'],
#            [1,0,1,0,'f','$$$',0,1,'f','>60'],
#            [0,1,0,1,'s','$$',1,1,'i','0-10'],
#            [0,1,0,0,'n','$',1,0,'b','0-10'],
#            [0,0,0,1,'s','$$',1,1,'t','0-10'],
#            [0,1,1,0,'f','$',1,0,'b','>60'],
#            [1,1,1,1,'f','$$$',0,1,'i','10-30'],
#            [0,0,0,0,'n','$',0,0,'t','0-10'],
#            [1,1,1,1,'f','$',0,0,'b','30-60']]
# y_train = [1,0,1,1,0,1,0,1,0,0,0,1]
#
# X_test = [[1,1,0,1,'s','$$',0,1,'t','10-30']]
#
# y_test = [[1]]

# print(features)
# print(labels)
# print(X_test)
# print(y_test)

### REAL DEAL
#load data
X_train, y_train, X_test, y_test = data.load_decision_tree_data()
# print(X_test)
# print(y_test)
X_train0 = [item[0] for item in X_train]
X_test0 = [item[0] for item in X_test]
print("unique values of Xtrain[0]",np.unique(X_train0))
print("unique values of ytrain",np.unique(y_train).tolist())
print("unique values of Xtest[0]",np.unique(X_test[0]).tolist())
print("unique values of ytest",np.unique(y_test).tolist())

# set classifier
dTree = decision_tree.DecisionTree()

# training
dTree.train(X_train.tolist(), y_train.tolist())
# dTree.train(X_train, y_train)

# print
Utils.print_tree(dTree)

import json
# testing
y_est_test = dTree.predict(X_test)
test_accu = accuracy_score(y_est_test, y_test)
print('test_accu', test_accu)

# print(np.unique(X_train[0]).tolist())
# print(np.unique(y_train).tolist())
# print(np.unique(X_test[0]).tolist())
# print(np.unique(y_test).tolist())
#
# print(np.unique(X_train))
# print(np.unique(y_train))

# # set classifier
# dTree = decision_tree.DecisionTree()
#
# # training
# dTree.train(X_train, y_train)
#
# # print
# Utils.print_tree(dTree)
# #
# import json
# # testing
# root_node = dTree.root_node
# print(root_node)
# print(root_node.splittable)
#
# featureUnique_split = dTree.root_node.featureUnique_split
# print(featureUnique_split)
#
# dim_split = dTree.root_node.dim_split
# print(dim_split)
#
# indexChildrenBranch = featureUnique_split.index(X_test[dim_split])
# print(indexChildrenBranch)
#
# print(dTree.root_node.children)
#
# y_est_test = dTree.predict(X_test)
# test_accu = accuracy_score(y_est_test, y_test)
# print('test_accu', test_accu)


# EXAMPLE: features with num_features * num_attributes (3*4)
# features = [[1,1,0,0],[1,1,1,0],[0,0,0,1]]
# branches = [[1,1],[1,0]] # Branches with num_branches * num_classes (3*4)
# labels = [1,1,0]
#
# count_max = 0
# for label in np.unique(labels):
#     if labels.count(label) > count_max:
#         count_max = labels.count(label)
#         # get the label that corresponds to max classes
#         cls_max = label
# print(cls_max)
#
# if len(np.unique(labels)) < 2:
#     splittable = False
# else:
#     splittable = True
# print(splittable)
#
# # splitable is false when there is no more attribute to be selected
# features_np = np.array(features)
# print(features_np.shape[0])
# print(features_np.shape[1])
# print(features_np[:,2])
# if features_np.shape[1] == 0:
#     splittable = False


# PRUNING
Utils.reduced_error_prunning(dTree, X_test, y_test)

y_est_test = dTree.predict(X_test)
test_accu = accuracy_score(y_est_test, y_test)
print('test_accu', test_accu)
Utils.print_tree(dTree)