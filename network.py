import numpy as np
import pandas as pd

def init_params():
    """
    Creates initial weights.
    
    Returns:
        W1 (np.array) - the multipling weights for the the first layer 
        b1 (np.array) - the additive weights for the the first layer
        W2 (np.array) - the multipling weights for the the second layer
        b2 (np.array) - the additive weights for the the first layer
    """
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    """
    Creates rectified liniar unit out of Z[1].

    Returns:
    (np.array) - rectified liniar unit for first layer. 
    """
    return np.maximum(Z, 0)

def softmax(Z):
    """
    Creates softmax of Z[2].
    
    Args: 
        Z (np.array) - the weighted value of Z from the second layer. 

    Returns:
        A (np.array) - softmax for the second layer. 
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propagation(W1, b1, W2, b2, X):
    """
    Implements forward propagation.
    
    Args:
        W1 (np.array) - the multipling weights for the the first layer 
        b1 (np.array) - the additive weights for the the first layer
        W2 (np.array) - the multipling weights for the the second layer
        b2 (np.array) - the additive weights for the the first layer

    Returns:
        Z1 (np.array) - the weighted value of Z from the first layer 
        A1 (np.array) - the rectified liniar unit for first layer layer
        Z2 (np.array) - the weighted value of Z from the second layer 
        A2 (np.array) - softmax for the second layer
    """
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_derivative(Z):
    """
    Takes the fisrst derivative of ReLU.
    
    Args:
        Z(np.array) - the rectified liniar unit for first layer layer
    Returns:
        Z (np.array) - the derivative for the Z from the first layer for the back propagation
    """
    return Z > 0

def one_hot(Y):
    """
    Implements one hot encoding.
    
    Args:
        Y(np.array) - labeled data

    Returns:
        Y (np.array) - encoded labeled data
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_propagation(Z1, A1, A2, W2, X, Y):
    """
    Implements backward propagation.

    Args:
        Z1 (np.array) - the weighted value of Z from the first layer 
        A1 (np.array) - the rectified liniar unit for first layer layer 
        A2 (np.array) - softmax for the second layer
        W2 (np.array) - the multipling weights for the the second layer

    Returns:
        dW1 (np.array) - the multipling weights for the the first layer for backward propagation
        db1 (np.array) - the additive weights for the the first layer for backward propagation
        dW2 (np.array) - the multipling weights for the the second layer for backward propagation
        db2 (np.array) - the additive weights for the the first layer for backward propagation
    """
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, eta):
    W1 = W1 - eta * dW1
    b1 = b1 - eta * db1
    W2 = W2 - eta * dW2
    b2 = b2 - eta * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descending(X, Y, eta, iterations):
    """ 
    Implements gradient descending and prints the resulting accuracy.

    Args:
        X (np.array) - input data
        Y (np.array) - labeled data
        eta (float) - hyperparametre for the descending
        iterations(int) - number of iterations
    
    Returns:
        W1 (np.array) - the multipling weights for the the first layer 
        b1 (np.array) - the additive weights for the the first layer
        W2 (np.array) - the multipling weights for the the second layer
        b2 (np.array) - the additive weights for the the first layer
    """
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 =  forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2,  b2 = update_params(W1,b1, W2, b2, dW1, db1, dW2, db2, eta)
        if i %  50 == 0:
            print('iteration: ', i )
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, Y))
    return W1, b1, W2, b2

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255. # Normalize the data

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255. # Normalize the data
_,m_train = X_train.shape



W1, b1, W2, b2 = gradient_descending(X_train, Y_train, 0.1, 500)
