import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        # Set initial value for iteration
        iter = 1

        # include the bias term in the weight vector at the end
        # wNew = np.insert(w, 0, b)
        wNew = np.insert(w, len(w), b)
        # print("wNew", wNew)

        # Append 1s to the training features
        # XNew = np.insert(X, 0, 1, axis=1)
        XNew = np.insert(X, len(X[0]), 1, axis=1)
        # print("XNew", XNew)

        # replace labels of 0 with 1 to be consistent with perceptron theme
        yNew = [item if item == 1 else -1 for item in y]
        # print("yNew", yNew)


        while iter < max_iterations + 1:
            # intialize initial update amount
            wInc = 0

            # initialize predicted labels
            yPred = np.zeros(N)

            # initialize counters:
            numSamples = 0

            # For each data point:
            for i, rowXNew in enumerate(XNew):
                if np.dot(wNew, rowXNew) > 0:  # row X already has 1s in the beginning to account for biases
                    yPred[i] = 1
                else:
                    yPred[i] = -1

                # print("yPred{i]", yPred[i])
                # print("yNew[1]",yNew[i])
                if yNew[i] * yPred[i] <= 0:  # meaning perceptron makes an error (y*y~<0)
                    # Update "incremental contribution" of each data point to weight vector: w(k+1) = w(k)+yn*xn
                    wInc = wInc + rowXNew * yNew[i]

                    numSamples += 1
            # print("wInc", wInc)

            # After visiting all data points, update the weight vector with average incremental weight update
            wIncAvg = step_size * wInc/N
            # print("wIncAvg", wIncAvg)

            wNew += wIncAvg
            # print("wNew of Perceptron", wNew)
            # update iteration counter
            iter += 1

        # obtain final bias and weight
        # b = wNew[0]
        b = wNew[-1]
        # print("b of perceptron",b)

        # w = wNew[1:]
        w = wNew[:-1]
        # print("w of perceptron",w)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #

        # loop through iterations
        for i in range(0, max_iterations):
            # use sigmoid to calculate predicted labels
            yPred = sigmoid(np.dot(X, w) + b)

            # calculate errors
            error = yPred - y

            # calculate derivatives
            derivatives = np.dot(X.T, error)

            # calculate new weight
            w -= (step_size / N) * derivatives
            b -= (step_size / N) * np.sum(error)

        assert w.shape == (D,)
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1/(1 + np.exp(-z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)

        # include the bias term in the weight vector
        # wNew = np.insert(w, 0, b)
        wNew = np.insert(w, len(w), b)

        # Append 1s to the training features
        # XNew = np.insert(X, 0, 1, axis=1)
        XNew = np.insert(X, len(X[0]), 1, axis=1)

        # loop through data point
        for i, rowX in enumerate(XNew):
            if np.dot(wNew, rowX) > 0:    # row X already has 1s in the beginning to account for biases
                preds[i] = 1
            else:
                preds[i] = 0

        # return preds
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        # preds = np.zeros(N)
        # include the bias term in the weight vector
        wNew = np.insert(w, 0, b)

        # Append 1s to the training features
        XNew = np.insert(X, 0, 1, axis=1)

        # calculate prediction
        # preds = sigmoid(np.dot(X, w) + b)
        preds = sigmoid(np.dot(XNew, wNew))

        # classify based on values of labels
        preds = np.asarray(preds > 0.5).astype(int)
        """"""

        """
        TODO: add your code here
        """
        # assert preds.shape == (N,)
        # return preds
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,)
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #


        # loop until max-iteration is achieved
        for iter in range(max_iterations):
            # get a random batch of samples
            idx = np.random.randint(N, size=1)
            Xnew = X[idx,:]
            ynew = y[idx]
            
            alpha = np.dot(Xnew, w.T) + b
            alpha -= alpha.max(-1, keepdims=True)
            
            
            yPred = np.exp(alpha)  
            yPred = yPred/yPred.sum(-1, keepdims=True)

            # yl = convert_to_onehot(y, C)
            Nnew = ynew.shape[0]
            yTrue = np.zeros([Nnew, C])
            yTrue[np.arange(Nnew), ynew.astype(int)] = 1.0

            error = yPred - yTrue
            derivatives = np.dot(error.T, Xnew)
            w -= (step_size / Nnew) * derivatives
            b -= (step_size / Nnew) * error.sum(0)

        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #

        # initiate weight matrix and biases vector
        w = np.zeros((C, D))
        b = np.zeros(C)

        # loop until max-iteration is achieved
        for it in range(max_iterations):
            alpha = np.dot(X, w.T) + b
            alpha -= alpha.max(-1, keepdims=True)

            yPred = np.exp(alpha)
            yPred = yPred / yPred.sum(-1, keepdims=True)

            # initialize
            yTrue = np.zeros([N, C])
            yTrue[np.arange(N), y.astype(int)] = 1.0

            # calculate error
            error = yPred - yTrue

            # calculate derivatives
            derivatives = np.dot(error.T, X)

            # update weights
            w -= (step_size / N) * derivatives

            # update biases
            b -= (step_size / N) * error.sum(0)
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape

    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    # preds = np.zeros(N)
    preds = np.dot(X, w.T) + b
    preds = np.argmax(preds, -1).T
    ############################################

    assert preds.shape == (N,)
    return preds




        