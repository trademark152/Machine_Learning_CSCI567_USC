# CSCI567:
This is a readme file for all programming assignments of CSCI 567 (Machine Learning) at the University of Southern California
for Lennard-Jones potential.

![Machine Learning](/graphic/ml.jpg)

## 0. Prerequisites
Python is the programming language used here
## 1. How to navigate and run

## 2. Files
### 1) HW1: KNN and Decision Tree
- Task 1: K-nearest neighbors

I initialize the KNN class with hyperparameters k and method of calculating distance. Min-max scaling and normalization are also performed during data preprocessing
![KNN](/graphic/knn.jpg)

- Task 2: Decision tree

I coded an object named DecisionTree with the following attributes: node object, feature dimension and functions: train, predict.
![Decision Tree](/graphic/decision_tree.jpg)

### 2) HW2: Regression and Classification
- Task 1: Regression

I implemented Linear and Polynomial Regression from scratch. Regularization is included as part of the package. My code is also capable of dealing with non-invertible matrix.

![Machine Learning](/graphic/regression.png)

- Task 2: Classification

I implemented both binary and multiclass classification. Firstly, binary classification with different loss functions (Perceptron and Logistic) is implemented. Secondly, multiclass classification (Stochastic gradient descent and gradient descent) is coded and different methods' performances are compared. Matrix computation and One-hot representation are utilized to optimize the performance.
![Classification](/graphic/classification.png)

### 3) HW3: Multilayer Perceptron and Neural Network

I implemented neural networks via a multilayer perceptron architecture. I use this neural network to classify MNIST database of handwritten digits (0-9). Cross-entropy loss is coded to handle k-class classification problem. I perform error-backpropagation by devising a way to compute partial derivatives (or gradients) w.r.t the parameters of a neural network, and use gradient-based optimization to learn the parameters.
![MLP](/graphic/mlp.jpg)

### 4) HW4: K-means and HMM
- Task 1: K-means clustering

4 minitasks are performed:
  <ul>
    <li> Implement K-means clustering algorithm to identify clusters in a two-dimensional toy-dataset. </li>
    <li> Implement image compression using K-means clustering algorithm </li>
    <li> Implement classification using the centroids identified by clustering on digits dataset </li>
    <li> Implement K-means++ clustering algorithm to identify clusters in a two-dimensional toy-dataset </li>
  </ul>
  ![kmeans](/graphic/kmeans.png)

- Task 2: Hidden Markov Model

I implemented a HMM to solve Part-of-Speech Tagging problem. Two algorithms are used for evaluation problem: the forward algorithm or the backwards algorithm.  Based on the result of forward algorithm and backward algorithm, I calculate sequence probability and posterior probability. For decoding I implement the Viterbi algorithm.
![HMM](/graphic/hmm.png)
