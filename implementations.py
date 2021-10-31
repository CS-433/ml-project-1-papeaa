import numpy as np
from proj1_helpers import *


##################
# Least Squares #
#################


def compute_mse(y, tx, w):
    """
    Calculate the loss with MSE
    :param y: labels
    :param tx: features
    :param w: weights
    :return: loss
    """
    ypred = predict_labels(w, tx)
    e = y - ypred
    return 1/2*e.T.dot(e)/len(y)


def compute_gradient(y, tx, w):
    """
    Compute the gradient for MSE.
    :param y: labels
    :param tx: features
    :param w: weights
    :return: gradient
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e)/len(y)


def least_squaresGD(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm.
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate gamma
    :return: loss, weights
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad
    loss = compute_mse(y, tx, w)
    return loss, w


def compute_stoch_gradient(y, tx, w, batch_size):
    """
    Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
    :param y: labels
    :param tx: features
    :param w: weights
    :param batch_size: number of examples
    :return: stochastic gradient
    """
    A = np.random.permutation(len(y))[:batch_size]
    e = y[A] - tx[A].dot(w)
    return -tx[A].T.dot(e)/batch_size


def least_squaresSGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param batch_size: number of examples
    :param max_iters: maximum number of iterations
    :param gamma: learning rate gamma
    :return: loss, weights
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_stoch_gradient(y, tx, w, batch_size)
        w = w - gamma*grad
    loss = compute_mse(y, tx, w)
    return loss, w


def least_squares(y, tx):
    """
    Calculate the least squares solution using normal equations.
    :param y: labels
    :param tx: features
    :return: loss, weights
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return compute_mse(y, tx, w), w


def ridge_regression(y, tx, lambda_):
    """
    Ridge Regression using normal equations. (L2 regularization)
    :param y: label
    :param tx: features
    :param lambda_: hyperparameter for regularization
    :return: loss, weights
    """
    k = 2*len(y)*lambda_
    w = np.linalg.solve((tx.T).dot(tx)+k*np.eye(len(tx[0])), (tx.T).dot(y))
    return compute_mse(y, tx, w), w


#######################
# Logistic Regression #
######################


def sigmoid(t):
    """
    Compute sigmoid function to a given element. The entries of
    the element are bounded between 1e^{10} and -1e^2 to avoid 
    overflows.
    :param t: element on which the sigmoid function will be applied
    :return: \sigma(x)
    """
    t[t >= 1e10] = 1e10
    t[t <= -1e2] = -1e2
    return 1/(1 + np.exp(-t))


def compute_log_loss(y, tx, w):
    """
    Compute loss for logistic regression. 
    :param y: labels
    :param tx: features
    :param w: weights
    :return: loss
    """
    yh = tx.dot(w)
    t = sigmoid(yh)
    t[abs(t) <= 1e-10] = 1e-10
    t[abs(t) >= 1-1e-10] = 1-1e-10
    y[abs(y) <= 1e-10] = 0
    return -np.sum(y*np.log(t)+(1-y)*np.log(1-t))/len(y)


def predict_label_log(w, tx):
    """
    Prediction function for logistic regression.
    :param w: weights
    :param tx: features
    :return: predctions
    """
    yh = tx.dot(w)
    h = sigmoid(yh)
    return np.round(h)


def compute_log_gradient(y, tx, w):
    """
    Compute gradient for the logistic regression.
    :param y: labels
    :param tx: features
    :param w: weights
    :return: gradient
    """
    z = tx.dot(w)
    yh = sigmoid(z)
    return np.dot(tx.T, yh-y)


def logistic_regression(
        y, tx, initial_w, max_iters, gamma):
    """
    Compute logistic regression with gradient descent algorithm.
    :param y: labels
    :param tx: features
    :param inital_w: intial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate gamma
    :return: loss, weights
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_log_gradient(y, tx, w)
        w = w - gamma*grad
    loss = compute_log_loss(y, tx, w)
    return loss, w


def compute_log_stochastic_gradient(y, tx, w, batch_size):
    """
    Compute a stochastic gradient for logistic regression from just 
    few examples n and their corresponding y_n labels.
    :param y: labels
    :param tx: features
    :param w: weights
    :param batch_size: number of examples
    :return: stochastic gradient
    """
    A = np.random.permutation(len(y))[:batch_size]
    z = tx[A].dot(w)
    yh = sigmoid(z)
    return np.dot(tx[A].T, yh-y[A])


def logistic_regression_stochastic_gradient(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Compute logistic regression with stochastic gradient descent algorithm.
    :param y: labels
    :param tx: features
    :param inital_w: intial weights
    :param batch_size: number of examples
    :param max_iters: maximum number of iterations
    :param gamma: learning rate gamma
    :return: loss, weights
    """
    w = initial_w
    for i in range(max_iters):
        grad = compute_log_stochastic_gradient(y, tx, w, batch_size)
        if i % 1000 == 0:
            gamma *= 0.1
        w = w - gamma*grad
    loss = compute_log_loss(y, tx, w)
    return loss, w


def compute_reg_log(y, tx, w, lambda_):
    """
    Compute loss for regularized logistic regression
    :param y: labels
    :param tx: features
    :param lambda_: hyperparameter for regularization
    :return: loss
    """
    return compute_log_loss(y, tx, w) + lambda_*np.dot(w.T, w)


def compute_reg_log_gradient(y, tx, w, lambda_):
    """
    Compute gradient for the regualarized logistic regression.
    :param y: labels
    :param tx: features
    :param w: weights
    :param lambda_: hyperparameter for regularization
    :return: gradient
    """
    return compute_log_gradient(y, tx, w) + lambda_*w


def reg_logistic_regression(
        y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Compute gradient descent algorithm for regularized logistic regression.
    :param y: labels
    :param tx: features
    :param lambda_: hyperparameter for regularization
    :param intial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate gamma
    :return: loss, weights
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_reg_log_gradient(y, tx, w, lambda_)
        w = w - gamma*grad
    loss = compute_reg_log(y, tx, w, lambda_)
    return loss, w


def compute_reg_log_stochastic_gradient(y, tx, w, batch_size, lambda_):
    """
    Compute a stochastic gradient for regularized logistic regression 
    from just few examples n and their corresponding y_n labels.
    :param y: labels
    :param tx: features
    :param w: weights
    :param batch_size: number of examples
    :param lambda_: hyperparameter for regularization
    :return: stochastic gradient
    """
    return compute_log_stochastic_gradient(y, tx, w, batch_size) + lambda_*w


def reg_logistic_regression_stochatic_gradient(
        y, tx, lambda_, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm for regularized logistic regression.
    :param y: labels
    :param tx: features
    :param lambda_: hyperparameter for regularization
    :param initial_w: initial weights
    :param batch_size: number of examples
    :param max_iters: maximum number of iterations
    :param gamma: learning rate gamma
    :return: loss, weights
    """
    w = initial_w
    for i in range(max_iters):
        grad = compute_reg_log_stochastic_gradient(
            y, tx, w, batch_size, lambda_)
        if i % 1000 == 0:
            gamma *= 0.1
        w = w - gamma*grad
    loss = compute_reg_log(y, tx, w, lambda_)
    return loss, w


#########
# Bonus #
########


def conjugate_gradient_descent(
        y, tx, initial_w, batch_size, max_iters):
    """
    Compute conjugate gradient descent.
    :param y: labels
    :param tx: features
    :param intial_w: initalized weights
    :param batch_size: number of examples
    :param max_iters: maximum number of iterations
    :return: loss, weights
    """
    xk = initial_w
    gfk = compute_log_stochastic_gradient(y, tx, xk,batch_size)
    pk = -gfk
    sigma_3 = 0.01
    alpha = np.logspace(-12, 1, 40)
    for k in range(max_iters):
        deltak = np.dot(gfk, gfk)
        min_ = np.inf
        ind = 0
        for i in range(len(alpha)): 
            L = compute_log_loss(y,tx,xk+alpha[i]*pk)
            if L<= min_ :
                min_ = L
                ind = i 
        xkp1 = xk + alpha[ind]  * pk
        gfkp1 = compute_log_stochastic_gradient(y, tx, xkp1,batch_size)
        yk = gfkp1 - gfk
        beta_k = max(0, np.dot(yk, gfkp1) / deltak)
        pkp1 = -gfkp1 + beta_k * pk
        if np.dot(pk, gfk) > -sigma_3 * np.dot(gfk, gfk):
            print("not stable" , k,alpha[ind])
            #break
        if np.linalg.norm(xk -  xkp1)<1e-7 : 
            print("close" , k,alpha[ind])
            break 
        xk = xkp1.copy()
        pk = pkp1.copy()
        gfk = gfkp1.copy()
    loss = min_
    w = xk
    return loss, w
