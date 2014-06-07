# -*- coding: utf-8 -*-
"""
Created on Sat Jun 07 12:48:25 2014

@author: Max
"""

# In order to properly choose the range for parameter being tested, we need to
# do some preliminary testing.

import os
import numpy
import neurolab as nl
from sklearn import preprocessing

os.chdir("..\\data\\")
# load the test data into a matrix
test = numpy.loadtxt(open("abalone_test.csv", "rb"), delimiter=" ", skiprows=1)
# the final column represents the target and the columns that precede it
# reprsent the features
# the input files are divided into 3 columns on the end that represent the
# a 1-hot-encoding of the target class value and the other (preceding) columns
# are features
numTarget = 3
numCols = test.shape[1]
numRows = test.shape[0]
numFeatures = numCols - 3
# after adjusting for 0-based indexing, the columns up until the 3rd-to-last
# are features
testFeatures = test[:, 0:numFeatures].reshape(numRows, numFeatures)
testTarget = test[:, numFeatures: numCols].reshape(numRows, numTarget)

# load the training data into a matrix
train = numpy.loadtxt(open("abalone_train.csv", "rb"), delimiter=" ", skiprows=1)
# the final column represents the target and the columns that precede it
# reprsent the features
numCols = train.shape[1]
numRows = train.shape[0]
numFeatures = numCols - 3
trainFeatures = train[:, 0:numFeatures].reshape(numRows, numFeatures)
trainTarget = train[:, numFeatures: numCols].reshape(numRows, numTarget)

netMinMax = []
for featureNum in range(numFeatures):
    netMinMax.append([min(trainFeatures[:, featureNum]),
                      max(trainFeatures[:, featureNum])])
# print "Min and Max Values:"
#print netMinMax

numHiddenLayers = [1, 2, 3]
hiddenLayerSize = [5, 10]
net1 = nl.net.newff(netMinMax, [5, 5, numTarget])
errors = []
for j in range(len(numHiddenLayers)):
    for i in range(len(hiddenLayerSize)):
        layerNums = [hiddenLayerSize[i]] * numHiddenLayers[j]
        layerNums.append(numTarget)
        print layerNums
        net = nl.net.newff(netMinMax, layerNums)
        error = net.train(trainFeatures, trainTarget, epochs=2, show=1, goal=1)
