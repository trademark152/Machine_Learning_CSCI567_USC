import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        # initialize classifier name and root node
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init

        # check that there exists features
        assert (len(features) > 0)

        # get the dimension of feature input
        self.feature_dim = len(features[0])

        # get the number of classes
        num_cls = np.unique(labels).size

        # build the tree by running TreeNode with given features, labels and number of classes
        self.root_node = TreeNode(features, labels, num_cls)

        # check if root node is splittable
        if self.root_node.splittable:
            self.root_node.split()
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []

        # loop through each index and each feature
        for idx, feature in enumerate(features):
            # obtain predicted value by running prediction through the root_node
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], children: [TreeNode], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.prediction_label = None

        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                # get the label that corresponds to max classes
                self.cls_max = label

        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        # raise NotImplementedError
        # initialize information gain
        IG_gain_max = -1e5

        # if we have no more examples, return majority label of parent.
        if len(self.labels) == 0:
            self.prediction_label = self.cls_max
            return

        # if not splittable, return majority label
        if not self.splittable:
            self.prediction_label = self.cls_max
            return

        # get initial entropy of the root node
        labelUniques, counts = np.unique(self.labels, return_counts=True)
        IG_init = Util.get_entropy(counts)

        labelUniques = labelUniques.tolist()
        print("Label Unique", labelUniques)

        # splittable is false when there is no more attribute to be selected
        featuresArray = np.array(self.features)
        print("featuresArray", featuresArray)
        if featuresArray.shape[1] == 0:
            self.splittable = False

        # loop through all feature's attributes (i.e dimension)
        for indexDimension in range(featuresArray.shape[1]):
            print("indexDimension to split", indexDimension)
            # select that attribute
            feature = featuresArray[:, indexDimension]

            # gather all possible unique values within that attribute, add to a list
            featureUnique = np.unique(feature).tolist()
            print("featureUnique", featureUnique)

            # get the number of branch based on this list of unique features
            num_branch = len(featureUnique)

            # if only 1 branch is possible then we are done splitting
            if num_branch < 2:
                continue

            # initiate individual branches
            # branches = [branch1, branch2,...] = [[class1_count, class2_count,...], [class1_count, class2_count,...]]
            branches = []
            # loop through each branch
            for idx in range(num_branch):
                branches.append([0] * self.num_cls)
            print("Before count", branches)

            # count the occurrences of each class in each branch
            for index_element, element in enumerate(feature):
                # # if new feature is in training data
                # if element in featureUnique:
                # signify branch to add counts:
                index_branch = featureUnique.index(element)
                # print()
                # print("branch to add to", index_branch)

                # signify class to add counts:
                index_class = labelUniques.index(self.labels[index_element])
                # print("class to add to", self.labels[index_element])


                # add it to correct locations
                branches[index_branch][index_class] += 1
                # print("After count", branches)
            print("After count", branches)
            print()

            # check for branch that has maximum information gain
            IG_branch = Util.Information_Gain(IG_init, branches)
            if IG_branch > IG_gain_max:
                IG_gain_max = IG_branch
                self.dim_split = indexDimension
                self.featureUnique_split = featureUnique

            # Tie-breaking
            if IG_branch == IG_gain_max:
                #  when there is a tie of information gain when comparing the attributes,
                # always choose the attribute which has more attribute values.
                if len(featureUnique) > len(self.featureUnique_split):
                    IG_gain_max = IG_branch
                    self.dim_split = indexDimension
                    self.featureUnique_split = featureUnique
                else:
                    # If they are all same, use the one with small index.
                    if indexDimension < self.dim_split:
                        IG_gain_max = IG_branch
                        self.dim_split = indexDimension
                        self.featureUnique_split = featureUnique


        # if information gain is unchanged
        if IG_gain_max == -1e5:
            self.splittable = False
            return

        print("IG_gain_max", IG_gain_max)
        print("best dim_split", self.dim_split)
        print("featureUnique_split", self.featureUnique_split)
        print()


        # SPLITTING
        # loop through each branch
        for idx_branch in range(len(self.featureUnique_split)):
            # initiate children features and labels
            child_features = []
            child_labels = []

            # loop through each feature
            for idx_element in range(len(self.features)):

                # split from dim_split
                if featuresArray[idx_element, self.dim_split] == self.featureUnique_split[idx_branch]:
                    # remove feature off the original array
                    new_feature_np = np.concatenate((featuresArray[idx_element, :self.dim_split],
                                                     featuresArray[idx_element, self.dim_split + 1:]))

                    # add to children features and labels
                    child_features.append(new_feature_np.tolist())
                    child_labels.append(self.labels[idx_element])

            # if splitting does not work
            if len(child_labels) == len(self.labels):
                self.splittable = False
                return

            print("Child features: ",child_features)
            print("Child labels: ",child_labels)

            # create child node
            child = TreeNode(child_features, child_labels, self.num_cls)

            # if children nodes can be splitted
            self.children.append(child)
            if child.splittable:
                child.split()
        return

        # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        # raise NotImplementedError
        # featureArray = np.array(feature)

        # if the node is still splittable
        if self.splittable:
            # gather all possible unique values within that attribute, add to a list
            # featureUnique = np.unique(feature).tolist()
            # If fall into the case no example on this feature, return the label of his parent node (majority class)
            if feature[self.dim_split] not in self.featureUnique_split:
                return self.cls_max
            else:
                # obtain index of children branch that test data belongs based on the corresponding unique value
                indexChildrenBranch = self.featureUnique_split.index(feature[self.dim_split])

            # extract refined feature by removing the attribute used for splitting
            # feature = np.concatenate(feature[:self.dim_split], feature[self.dim_split + 1:])
            # print(self.dim_split)
            # print(feature[:self.dim_split])
            # print(feature[self.dim_split + 1:])
            if type(feature) is list:
                feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            else:
                feature = np.asarray(feature[:self.dim_split].tolist() + feature[self.dim_split + 1:].tolist())
            # print(feature)

            # recursively do prediction on children nodes
            return self.children[indexChildrenBranch].predict(feature)

        # if not splittable
        else:
            return self.cls_max
