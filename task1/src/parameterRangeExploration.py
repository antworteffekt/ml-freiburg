# -*- coding: utf-8 -*-
"""
Created on Sat Jun 07 12:48:25 2014

@author: Max
"""

# In order to properly choose the range for parameter being tested, we need to
# do some preliminary testing.

import os
import numpy as np
import neurolab as nl
from sklearn import preprocessing


def loadAbalone():
    "This function loads the Abalone data into training and test matrices."
    # set the working directory to the data folder
    os.chdir("..\\data\\")
    # load the test data into a matrix. the file is space delimeted and has a header row.
    test = np.loadtxt(open("abalone_test.csv", "rb"), delimiter=" ", skiprows=1)
    # the final column represents the target and the columns that precede it represent the features
    # the input files are divided into 3 columns on the end that represent the a 1-hot-encoding of the target class value
    # and the other (preceding) columns are features
    numTarget = 3
    numRows, numCols = test.shape
    numFeatures = numCols - 3
    # after adjusting for 0-based indexing, the columns up until the 3rd-to-last are features
    testFeatures = test[:, 0:numFeatures].reshape(numRows, numFeatures)
    testTarget = test[:, numFeatures: numCols].reshape(numRows, numTarget)

    # load the training data into a matrix. the file is space delimeted and has a header row.
    train = np.loadtxt(open("abalone_train.csv", "rb"), delimiter=" ", skiprows=1)
    # the final column represents the target and the columns that precede it represent the features
    numRows, numCols = train.shape
    numFeatures = numCols - 3
    trainFeatures = train[:, 0:numFeatures].reshape(numRows, numFeatures)
    trainTarget = train[:, numFeatures: numCols].reshape(numRows, numTarget)
    return {'trainFeatures': trainFeatures, 'trainTarget': trainTarget, 'testFeatures': testFeatures,
            'testTarget': testTarget}


def loadSineData():
    "This function generates matrices sufficient to test a neural network, using a sine function."
    trainFeaturesRaw = np.linspace(-7, 7, 20)
    trainTargetRaw = np.sin(trainFeaturesRaw) * .5
    size = len(trainFeaturesRaw)
    trainFeatures = trainFeaturesRaw.reshape(size, 1)
    trainTarget = trainTargetRaw.reshape(size, 1)

    testFeaturesRaw = np.linspace(-6.0, 6.0, 150)
    testTargetRaw = np.sin(testFeaturesRaw) * .5
    testFeatures = testFeaturesRaw.reshape(len(testFeaturesRaw), 1)
    testTarget = testTargetRaw.reshape(len(testFeaturesRaw), 1)
    return {'trainFeatures': trainFeatures, 'trainTarget': trainTarget, 'testFeatures': testFeatures,
            'testTarget': testTarget}


data = loadSineData()
trainFeatures, trainTarget, testFeatures, testTarget = data['trainFeatures'], data['trainTarget'], \
                                                       data['testFeatures'], data['testTarget']

assert len(trainTarget.shape) == len(testTarget.shape)
assert len(trainFeatures.shape) == len(testFeatures.shape)
netMinMax = []
if len(trainFeatures.shape) > 1:
    numFeatures = trainFeatures.shape[1]
    assert trainTarget.shape[1] == testTarget.shape[1]
    for featureNum in range(trainFeatures.shape[1]):
        netMinMax.append([min(trainFeatures[:, featureNum]),
                          max(trainFeatures[:, featureNum])])
else:
    numFeatures = 1
    netMinMax.append([min(trainFeatures),
                      max(trainFeatures)])

if len(trainTarget.shape) > 1:
    numTarget = trainTarget.shape[1]
    assert trainTarget.shape[1] == testTarget.shape[1]

else:
    numTarget = 1

numHiddenLayers = [0, 1, 2]
hiddenLayerSize = [5, 10]
net1 = nl.net.newff(netMinMax, [5, 5, numTarget])
errors = {}
for j in numHiddenLayers:
    for i in hiddenLayerSize:
        if j == 0 and i == hiddenLayerSize[0] or j > 0:
            layerNums = [i] * j
            layerNums.append(numTarget)
            print layerNums
            net = nl.net.newff(netMinMax, layerNums)
            # this returns the entire history of errors vs. epochs
            error = net.train(trainFeatures, trainTarget, epochs=100, show=10, goal=.02)
            # however, we are only interested in the final error
            errors[(i, j)] = error[-1]
print errors