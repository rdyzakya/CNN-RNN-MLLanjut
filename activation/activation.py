import numpy as np

clipping_point = 500

def set_clipping_point(value : int) -> None:
    """
    [DESC]
        Set the clipping point for the sigmoid function
    [PARAMS]
        value : int
            New clipping point
    """
    global clipping_point
    clipping_point = value

def linear(x : np.ndarray, derivative=False) -> np.ndarray:
    """
    [DESC]
        Linear activation function
    [PARAMS]
        x : np.ndarray
            Input
    [RETURN]
        np.ndarray
            Output
    """
    if derivative:
        return np.ones_like(x)
    return x

def sigmoid(x : np.ndarray, derivative=False) -> np.ndarray:
    """
    [DESC]
        Sigmoid activation function
    [PARAMS]
        x : np.ndarray
            Input
    [RETURN]
        np.ndarray
            Output
    """
    x = np.clip(x, -clipping_point, clipping_point)
    p = 1 / (1 + np.exp(-x))
    if derivative:
        return p * (1 - p)
    return p

def relu(x : np.ndarray, derivative=False) -> np.ndarray:
    """
    [DESC]
        ReLU activation function
    [PARAMS]
        x : np.ndarray
            Input
    [RETURN]
        np.ndarray
            Output
    """
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def softmax(x : np.ndarray) -> np.ndarray:
    """
    [DESC]
        Softmax activation function
    [PARAMS]
        x : np.ndarray
            Input
    [RETURN]
        np.ndarray
            Output
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def tanh(x : np.ndarray) -> np.ndarray:
    """
    [DESC]
        Tanh activation function
    [PARAMS]
        x : np.ndarray
            Input
    [RETURN]
        np.ndarray
            Output
    """
    return np.tanh(x)

def leaky_relu(x : np.ndarray) -> np.ndarray:
    """
    [DESC]
        Leaky ReLU activation function
    [PARAMS]
        x : np.ndarray
            Input
    [RETURN]
        np.ndarray
            Output
    """
    return np.maximum(0.01 * x, x)

def softplus(x : np.ndarray) -> np.ndarray:
    """
    [DESC]
        Softplus activation function
    [PARAMS]
        x : np.ndarray
            Input
    [RETURN]
        np.ndarray
            Output
    """
    return np.log(1 + np.exp(x))