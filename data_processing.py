import numpy as np
from implementations import *


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio.
    :param x: features
    :param y: labels
    :param ration: given ration for splitting dataset
    :param seed: seed for permutation, default: 1
    :return: Training Features, Test Features, Training Labels, Test Labels
    """
    np.random.seed(seed)
    n = len(x)
    p = np.random.permutation(np.arange(n))
    a = int(n*ratio)
    xtrain = x[p[:a]]
    xtest = x[p[a:]]
    ytrain = y[p[:a]]
    ytest = y[p[a:]]
    return xtrain, xtest, ytrain, ytest


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.
    :param y: labels
    :param k_fold: number foldings
    :param seed: seed for permutatio
    :return: k-indices
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: features
    :param degree: degree of polynomial basis
    :return: feature expended featureset
    """
    M = np.ones((len(x), len(x[0])*degree+1))
    for j in range(1, degree+1):
        M[:, (len(x[0])*(j-1)+1):(len(x[0])*j+1)] = x**j
    return M


def build_advanced_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree."""
    M = np.ones((len(x), 2*(len(x[0])*degree)+1))
    print(M.shape)
    for j in range(1, degree+1):
        M[:, (len(x[0])*(j-1)+1):(len(x[0])*j+1)] = x**j
    for j in range(degree+1, 2*degree+1):
        p = np.random.permutation(len(x[0]))
        m = np.random.randint(1, degree+1)
        n = np.random.randint(1, degree+1)
        M[:, (len(x[0])*(j-1)+1):(len(x[0])*j+1)] = x**m*x[:, p]**n
    return M


def fix_nonexisting(tX, tX_test):
    """
    Replace -999 by median of the datapoint. 
    No return, because it's operating directly on the reference.
    :param tX: training features
    :param tX_test: test features
    :return: training feature, test feature
    """
    data = np.vstack([tX, tX_test])  # To have unified distribution
    data[data == -999] = np.NaN
    col_median = np.nanmedian(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_median, inds[1])
    return data[:tX.shape[0]], data[tX.shape[0]:]


def standardize(tX, tX_test):
    """
    Standardize Features
    :param tX: training features
    :param tX_test: test features
    :return: training feature, test feature
    """
    data = np.vstack([tX, tX_test])  # To have unified distribution
    a = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    data = (data-a)/s
    return data[:tX.shape[0]], data[tX.shape[0]:]


def cross_validation_least_degree(y, x, k_indices, k, degree):
    """
    Cross Validation for degrees of polynomial for feature expension
    for Least Squares.
    :param y: labels
    :param x: features
    :param k_indices: k indices
    :param k: index of selected k indice
    :param degree: degree for polynomial for feature expansion
    :return: loss for training, loss for test
    """
    test = k_indices[k]
    nx = build_poly(x, degree)
    train = np.reshape(np.delete(k_indices, k, 0),
                       (len(k_indices)-1)*len(test))
    xtrain = nx[train]
    xtest = nx[test]
    ytrain = y[train]
    ytest = y[test]
    loss_tr, w_train = least_squares(ytrain, xtrain)
    return loss_tr, compute_mse(ytest, xtest, w_train)


def cross_validation_ridge(y, x, k_indices, k, lambda_):
    """
    Cross Validation for lambdas for Ridge Regression.
    :param y: labels
    :param x: features
    :param k_indices: k indices
    :param k: index of selected k indice
    :param lambda_: hyperparameter for regularization
    :return: loss for training, loss for test
    """
    test = k_indices[k]
    train = np.reshape(np.delete(k_indices, k, 0),
                       (len(k_indices)-1)*len(test))
    xtrain = x[train]
    xtest = x[test]
    ytrain = y[train]
    ytest = y[test]
    loss_tr, w_train = ridge_regression(ytrain, xtrain, lambda_)
    return loss_tr, compute_mse(ytest, xtest, w_train)


def cross_validation_log_degree(y, x, k_indices, k, degree):
    """
    Cross Validation for degrees of polynomial for feature expension
    for Logistic Regression.
    :param y: labels
    :param x: features
    :param k_indices: k indices
    :param k: index of selected k indice
    :param degree: degree for polynomial for feature expansion
    :return: loss for training, loss for test
    """
    test = k_indices[k]
    nx = build_poly(x, degree)
    train = np.reshape(np.delete(k_indices, k, 0),
                       (len(k_indices)-1)*len(test))
    xtrain = nx[train]
    xtest = nx[test]
    ytrain = y[train]
    ytest = y[test]
    batch_size = len(ytrain)//20
    initial_w = np.zeros(len(xtrain[0]))
    loss_tr, w_train = logistic_regression_stochastic_gradient(
        ytrain, xtrain, initial_w, batch_size, 500, 1e-5)
    return loss_tr, compute_log_loss(ytest, xtest, w_train)


def cross_validation_reg_logistic_lambda(y, x, k_indices, k, lambda_):
    """
    Cross Validation for lambdas for Regularized Logistic Regression.
    :param y: labels
    :param x: features
    :param k_indices: k indices
    :param k: index of selected k indice
    :param lambda_: hyperparameter for regularization
    :return: loss for training, loss for test
    """
    test = k_indices[k]
    train = np.reshape(np.delete(k_indices, k, 0),
                       (len(k_indices)-1)*len(test))
    xtrain = x[train]
    xtest = x[test]
    ytrain = y[train]
    ytest = y[test]
    batch_size = len(ytrain)//20
    initial_w = np.zeros(len(xtrain[0]))
    loss_tr, w_train = reg_logistic_regression_stochatic_gradient(
        ytrain, xtrain, lambda_, initial_w, batch_size, 500, 1e-5)
    return loss_tr, compute_log_loss(ytest, xtest, w_train)