import numpy as np

EPS = 1e-10

def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), EPS, 1 - EPS)

def f(w, x):
    return sigmoid(np.dot(x, w))

def BCE(pred_y, true_y):
    return -np.dot(true_y.T, np.log(pred_y)).item() - np.dot((1 - true_y).T, np.log(1 - pred_y)).item()

def accuracy(pred_y, true_y):
    return (np.dot(true_y.T, pred_y) + np.dot(1 - true_y.T, 1 - pred_y)).item() / true_y.shape[0]
