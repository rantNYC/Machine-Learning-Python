#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cvxopt
import scipy as sp
import time
# acquire data, split it into training and testing sets (50% each)
# nc -- number of classes for synthetic datasets
def acquire_data(data_name, nc = 2):
    if data_name == 'synthetic-easy':
        print ('Creating easy synthetic labeled dataset')
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 0 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-medium':
        print ('Creating medium synthetic labeled dataset')
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 3 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-hard':
        print ('Creating hard easy synthetic labeled dataset')
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 5 * rng.uniform(size=X.shape)
    elif data_name == 'moons':
        print ('Creating two moons dataset')
        X, y = datasets.make_moons(noise=0.2, random_state=0)
    elif data_name == 'circles':
        print ('Creating two circles dataset')
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        print ('Loading iris dataset')
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif data_name == 'digits':
        print ('Loading digits dataset')
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        print ('Loading breast cancer dataset')
        bcancer = datasets.load_breast_cancer()
        X = bcancer.data
        y = bcancer.target
    else:
        print ('Cannot find the requested data_name')
        assert False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# compare the prediction with grount-truth, evaluate the score
def myscore(y, y_gt):
    assert len(y) ==  len(y_gt)
    return np.sum(y == y_gt)/float(len(y))

# plot data on 2D plane
# use it for debugging
def draw_data(X_train, X_test, y_train, y_test, nclasses):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.jet
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()

# draw results on 2D plan for binary classification
# this is a fake version (using a random linear classifier)
# modify it for your own usage (pass in parameter etc)
def draw_result_binary_fake(X_train, X_test, y_train, y_test, alpha, bias):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot
    tmpX = np.c_[xx.ravel(), yy.ravel()]

    Z_class, Z_pred_val = get_prediction_fake(X_train, y_train, X_test, alpha, bias)

    Z_clapped = np.zeros(Z_pred_val.shape)
    Z_clapped[Z_pred_val>=0] = 1.5
    Z_clapped[Z_pred_val>=1.0] = 2.0
    Z_clapped[Z_pred_val<0] = -1.5
    Z_clapped[Z_pred_val<-1.0] = -2.0

    Z = Z_clapped.reshape(X_train.shape[0])

#    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha = .4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    #    ax = plt.figure(1)
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', marker='x', linewidth=3,
                label='Test Data')

    y_train_pred_class, y_train_pred_val = get_prediction_fake(X_train, y_train, X_test, alpha, bias)
    sv_list_bool = np.logical_and(y_train_pred_val >= -1.0, y_train_pred_val <= 1.0)
    sv_list = np.where(sv_list_bool)[0]
    plt.scatter(X_train[sv_list, 0], X_train[sv_list, 1], s=100, facecolors='none', edgecolors='orange', linewidths = 3, label='Support Vectors')

    y_test_pred_class, y_test_pred_val = get_prediction_fake(X_train, y_train, X_test, alpha, bias)
    score = myscore(y_test_pred_class, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

    plt.legend()
    plt.show()

# predict labels using a random linear classifier
# returns a list of length N, each entry is either 0 or 1
def get_prediction_fake(x, y, x_predict, alpha, bias):
    m, n = x_predict.shape
    y_predict = np.zeros(m)
    y_pred_class = np.zeros(m)
    for i in range(m):
        y_predict[i] = np.sum((alpha * np.vstack(y)).transpose() * np.dot(x_predict[i], x.transpose()).transpose())
        y_pred_class[i] = 1*(y_predict[i]>=0.0) 
    y_pred_val =  np.sign(y_predict + bias)
    
    return y_pred_class, y_pred_val
####################################################
# binary label classification

# training kernel svm
# return sv_list: list of surport vector IDs
# alpha: alpha_i's
# b: the bias
def mytrain_binary(X_train, y_train, C, ker, kpar):
    print ('Start training ...')
    nsample, nfeature = X_train.shape
    y_train = y_train.astype(float)
    sv_list= []
    b = []
    alpha = []
    
    P = cvxopt.matrix(np.multiply(kpar, np.outer(y_train,y_train)))
    q = cvxopt.matrix(np.ones(nsample) * -1)
    
    A = cvxopt.matrix(y_train, (1,nsample))
    b = cvxopt.matrix(0.0)
    
    G1 = np.diag(np.ones(nsample)*-1)
    G2 = np.diag(np.ones(nsample))
    G = cvxopt.matrix(np.vstack((G1, G2)))
    h1 = np.zeros(nsample)
    h2 = np.ones(nsample)*C
    h = cvxopt.matrix(np.hstack((h1, h2)))
    
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.ravel(solution['x'])
    
    sv = alpha > 1e-4
    ind = np.arange(len(alpha))[sv]
    sv_list = X_train[sv]
    alpha = alpha[sv]
    sv_y = y_train[sv]
    
    b = 0
    for i in range(len(alpha)):
       b += sv_y[i]
       b -= np.sum(alpha * sv_y * kpar[ind[i], sv])
    b /= len(alpha)
            
    print ('Finished training.')
    return sv_list, alpha, b

# predict given X_test data,
# need to use X_train, ker, kpar_opt to compute kernels
# need to use sv_list, y_train, alpha, b to make prediction
# return y_pred_class as classes (convert to 0/1 for evaluation)
# return y_pred_value as the prediction score
def mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar):
    
    m, n = X_train.shape #m - number of samples; n - number of features
    C, kpar = my_cross_validation(X_train, y_train, ker)
    sv, a, b = mytrain_binary(X_test, y_train, C, ker, kpar)
    y_pred_class, y_pred_value = get_prediction_fake(X_train, y_train, X_test, a, b)
            
    return y_pred_class, y_pred_value

# use cross validation to decide the optimal C and the kernel parameter kpar
# if linear, kpar = -1 (no meaning)
# if polynomial, kpar is the degree
# if gaussian, kpar is sigma-square
# k -- number of folds for cross-validation, default value = 5
def my_cross_validation(X_train, y_train, ker, k = 5):
    assert ker == 'linear' or ker == 'polynomial' or ker == 'gaussian'
    n_samples, n_features = X_train.shape
    kpar_opt = np.zeros((n_samples, n_samples))
    C_opt = 10
    if ker == 'linear':
        for i in range(k):
            kpar_opt = np.inner(X_train, X_train)
                
    if ker == 'polynomial':
        for i in range(k):
            kpar_opt = (1 + np.dot(X_train, X_train.transpose()))**X_train.shape[1]
            
    if ker == 'gaussian':
        for i in range(k):
            sigma = np.std(X_train)
            pairwise_dists = sp.spatial.distance.squareform(sp.spatial.distance.pdist(X_train, 'euclidean'))
            kpar_opt = np.exp(-pairwise_dists ** 2 / 2*sigma ** 2)
            
    return C_opt, kpar_opt

################

def main():

    #######################
    # get data
    # only use binary labeled

    X_train, X_test, y_train, y_test = acquire_data('synthetic-easy')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-medium')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-hard')
    # X_train, X_test, y_train, y_test = acquire_data('moons')
    # X_train, X_test, y_train, y_test = acquire_data('circles')
    # X_train, X_test, y_train, y_test = acquire_data('breast_cancer')

    nfeatures = X_train.shape[1]    # number of features
    ntrain = X_train.shape[0]   # number of training data
    ntest = X_test.shape[0]     # number of test data
    y = np.append(y_train, y_test)
    nclasses = len(np.unique(y)) # number of classes

    # only draw data (on the first two dimension)
    draw_data(X_train, X_test, y_train, y_test, nclasses)
    
    ker = 'linear'
    # ker = 'polynomial'
    # ker = 'gaussian'
    st = time.time()
    C_opt, kpar_opt = my_cross_validation(X_train, y_train, ker)
    et = time.time()
    difference = et - st
    print('Average running time', difference)
    
    st = time.time()
    sv_list, alpha, b = mytrain_binary(X_train, y_train, C_opt, ker, kpar_opt)
    et = time.time()
    difference = et - st
    print('Average running time', difference)

    # a fake function to draw svm results
    draw_result_binary_fake(X_train, X_test, y_train, y_test, alpha, b)
    
    st = time.time()
    y_test_pred_class, y_test_pred_val = mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar_opt)
    et = time.time()
    difference = et - st
    print('Average running time', difference)

    test_score = myscore(y_test_pred_class, y_test)

    print ('Test Score:', test_score)

if __name__ == "__main__": main()