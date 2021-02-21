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

    
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #               
        w=np.zeros(D)
        b=0
        y[y==0]=-1
        X=np.insert(X, 0, 1, 1)  
        wstar=np.zeros(D+1)       
        avg_lambda=(step_size/N)
        for i in range(max_iterations):
            wstar=wstar+(np.dot(np.where(y*(np.dot(X,wstar))<=0,1,0)*y,X))*avg_lambda
        w=wstar[1:]
        b=wstar[0]
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        wstar=np.zeros(D+1)
        X=np.insert(X,0,1,1)
        avg_lambda=(step_size/N)
        for i in range(max_iterations):
            scores= np.dot(X,wstar)
            y_preds=sigmoid(scores)
            
            #Update weights
            err=y-y_preds
            gradient=np.dot(X.transpose(),err)
            wstar += avg_lambda*gradient
        w=wstar[1:]
        b=wstar[0]
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
        summ = (np.dot(X,w) + b)
        preds=np.where(summ>0,1.,0.)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        summ = sigmoid(np.dot(X,w) + b)
        preds = np.where(summ>0.5,1.,0.)
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds

def converttoOneHot(vector, N, C):
    result=np.zeros((N,C))
    result[np.arange(N),vector]=1
    return result

def gd_softmax(z):
    e=np.exp(z)
    return (e / np.sum(e, axis=0))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

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
        w = np.zeros((C, D))
        b = np.zeros(C)
        wstar=np.zeros((C,D))
        y=converttoOneHot(y,N,C)
        for i in range(max_iterations):
            n=np.random.choice(N)
            py=softmax(np.dot(X[n],wstar.T)+b)
            dy=y[n]-py
            wstar-=step_size*-np.dot(dy.reshape(C,1),X[n].reshape(D,1).T)
            b+=step_size*dy
        w=wstar
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        step_size=step_size/N
        wstar=np.zeros((C,D+1))
        X=np.insert(X,0,1,1)
        y=converttoOneHot(y,N,C)
        for i in range(max_iterations):
            z=np.dot(wstar,X.T)
            z= z-np.max(z)
            py=gd_softmax(z)
            py=py-y.T
            mistake=np.dot(py,X)
            wstar-=step_size*mistake
        w=wstar[:,1:]
        b=wstar[:,0]

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
    preds = softmax(np.dot(X,w.T)+b)
    preds = np.argmax(preds, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




        