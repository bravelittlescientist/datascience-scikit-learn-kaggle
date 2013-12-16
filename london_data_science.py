#!/usr/bin/python2

# This file reads in a training and testing dataset, runs a trainer
# on the training data, and predicts the output data.

import sys
import csv
import os

import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

def train(Xtrain, Ytrain, Xtest, set_aside=False):
    """ Trains and predicts dataset with a SVM classifier """

    set_aside=True

    # Sometimes set aside some data for testing
    if set_aside:
        XtrainS, XtestS, YtrainS, YtestS = train_test_split(Xtrain, Ytrain, test_size=.1)
    else:
        XtrainS = Xtrain
        YtrainS = Ytrain

    # Initialize grid search parameters
    Cs = 10 ** np.arange(2,9)
    gammas = 10 ** np.arange(-5,4)
    param_grid = dict(gamma=gammas, C=Cs)

    # Search grid.
    cv = StratifiedKFold(y=YtrainS, n_folds=2)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(XtrainS, YtrainS)

    print("The best classifier is: ", grid.best_estimator_)

    # Report 10% training sample if set_aside is on
    if set_aside:
        print "Classifier performance on 10% data:",accuracy_score(YtestS, grid.predict(XtestS))

    # predict on test data
    Ytest = grid.predict(Xtest)
    return Ytest

def write_test_labels(Ytest, outfile="data/testLabels.csv"):
    """ Writes 9000 testing predictions to file """
    f = open(outfile, 'w')

    f.write('Id,Solution\n')
    count = 1
    for prediction in Ytest:
        f.write("%d,%d\n" % (count,prediction))
        count += 1

    f.close()

def read_datasets():
    """ Reads test and training csv files """
    # Open the data files
    # TODO setup error handling for this in case file not present
    train_data_file = open("data/train.csv")
    train_data_labels = open("data/trainLabels.csv")
    test_data_file = open("data/test.csv")

    # Read in CSV file
    Xtrain = np.array([map(float, row) for row in csv.reader(train_data_file)])
    Ytrain = np.array([int(row) for row in train_data_labels])
    Xtest = np.array([map(float, row) for row in csv.reader(test_data_file)])

    # Close the files like a boss
    train_data_file.close()
    train_data_labels.close()
    test_data_file.close()

    # Return Xtrain, Ytrain, and Xtest
    return Xtrain, Ytrain, Xtest

if __name__ == "__main__":
    # Argument handling for custom output files
    print "Data Science London - Scikit Learn Practice"
    print "http://www.kaggle.com/c/data-science-london-scikit-learn"
    print
    print "Usage: python london_data_science.py [Output]"

    predictionsFile = "data/testLabels.csv"
    if len(sys.argv) == 2:
        predictionsFile = sys.argv[1]
        print "Will output predictions to user-specifiied file:",predictionsFile
    else:
        print "No output specified. Will write test predictions to data/testLabels.csv"
    print

    # This is a pretty small dataset - 1000 training, 9000 test.
    # It will load pretty quickly.
    Xtrain, Ytrain, Xtest = read_datasets()

    # Train random forest, predict result, write to output
    Ytest = train(Xtrain, Ytrain, Xtest)
    write_test_labels(Ytest, outfile=predictionsFile)
