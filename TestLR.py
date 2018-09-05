#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from LogisticRegression import acquire_data, mytrain_binary, mytest_binary, mytrain_multi, mytest_multi

def main():

    #######################
    # get data
    # binary labeled

    # X_train, X_test, y_train, y_test = acquire_data('synthetic-easy')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-medium')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-hard')
    # X_train, X_test, y_train, y_test = acquire_data('moons')
    # X_train, X_test, y_train, y_test = acquire_data('circles')
    # X_train, X_test, y_train, y_test = acquire_data('breast_cancer')

    # multi-labeled
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-easy', nc = 3)
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-medium', nc = 3)
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-hard', nc = 3)
    X_train, X_test, y_train, y_test = acquire_data('iris')
    # X_train, X_test, y_train, y_test = acquire_data('digits')


    nfeatures = X_train.shape[1]    # number of features
    ntrain = X_train.shape[0]   # number of training data
    ntest = X_test.shape[0]     # number of test data
    y = np.append(y_train, y_test)
    nclasses = len(np.unique(y)) # number of classes

    # only draw data (on the first two dimension)
    draw_data(X_train, X_test, y_train, y_test, nclasses)

    if nclasses == 2:
        st = time.time()
        w_opt = mytrain_binary(X_train, y_train)
        et = time.time()
        difference = et - st
        print('Average running time', difference)
        # debugging example
        draw_result_binary(X_train, X_test, y_train, y_test, w_opt)
    else:
        st = time.time()
        w_opt = mytrain_multi(X_train, y_train)
        et = time.time()
        difference = et - st
        print('Average running time', difference)

    if nclasses == 2:
        y_train_pred = mypredict_binary(X_train, w_opt)
        y_test_pred = mytest_binary(X_test, w_opt)
    else:
        y_train_pred = mypredict_multi(X_train, w_opt)
        y_test_pred = mypredict_multi(X_test, w_opt)

    train_score = myscore(y_train_pred, y_train)
    test_score = myscore(y_test_pred, y_test)

    print ('Training Score:', train_score)
    print ('Test Score:', test_score)

if __name__ == "__main__": main()