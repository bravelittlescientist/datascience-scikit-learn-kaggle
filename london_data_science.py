#!/usr/bin/python2

# This file reads in a training and testing dataset, runs a trainer
# on the training data, and predicts the output data.

import sys
import csv
import os

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score

def random_forest(Xtrain, Ytrain, Xtest):
    """ Trains and predicts dataset with a 600-member random forest

    Prints cross-validation information, as well.
    Returns Ytest, predictions for the test data."""

    print "Training a random forest..."
    clf = RandomForestClassifier(n_estimators = 600)

    print "Average cross-validation performance, 10-fold:"
    scores = cross_val_score(clf, Xtrain, Ytrain, cv=10)
    print scores.mean(),"+/-",scores.std()

    print "Predicting forest on test data..."
    clf.fit(Xtrain, Ytrain)
    Ytest = clf.predict(Xtest)
    return Ytest

def write_test_labels(Ytest, outfile="data/testLabels.csv"):
    """ Writes 9000 testing predictions to file """
    f = open(outfile, 'w')

    for prediction in Ytest:
        f.write(str(prediction) + "\n")

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
    Ytest = random_forest(Xtrain, Ytrain, Xtest)
    write_test_labels(Ytest, outfile=predictionsFile)
