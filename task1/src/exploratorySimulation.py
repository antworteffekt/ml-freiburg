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
import sys
from fractions import Fraction
from math import floor
import inspect as ins

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
numTrainingInstances = trainFeatures.shape[0]
numTestInstances = testFeatures.shape[0]
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
hiddenLayerSize = [1,2,3]
activationFunctions = {'log': nl.net.trans.LogSig(), 'linear': nl.net.trans.PureLin(), 'tan': nl.net.trans.TanSig()}
trainingMethods = {'gradientDescent': nl.net.train.train_gd, 'momentum': nl.net.train.train_gdm,
                   'adaptiveLearningRate': nl.net.train.train_gda, 'm+a': nl.net.train.train_gdx}

# Extract the default values for all of the parameters used by each sort of training method
gd = ins.getargspec(nl.net.train.gd.TrainGD.__init__)
m = ins.getargspec(nl.net.train.gd.TrainGDM.__init__)
alr = ins.getargspec(nl.net.train.gd.TrainGDA.__init__)
malr = ins.getargspec(nl.net.train.gd.TrainGDX.__init__)

# TODO: Consider other parameters for training algorithms
# Store the parameters for each sort of training method
# Currently, the set is determined by the default, +/- order of mag., +/- mult. of 2
trainingMethodParameters = {'gradientDescent': {'lr': [gd.defaults[0],
                                                       gd.defaults[0] * 10,
                                                       gd.defaults[0] / 10,
                                                       gd.defaults[0] * 2,
                                                       gd.defaults[0] / 2]},
                            'momentum': {'lr': [m.defaults[0],
                                                m.defaults[0] * 10,
                                                m.defaults[0] / 10,
                                                m.defaults[0] * 2,
                                                m.defaults[0] / 2]},
                            'adaptiveLearningRate': {'lr': [alr.defaults[0],
                                                            alr.defaults[0] * 10,
                                                            alr.defaults[0] / 10,
                                                            alr.defaults[0] * 2,
                                                            alr.defaults[0] / 2]},
                            'm+a': {'lr': [malr.defaults[0],
                                           malr.defaults[0] * 10,
                                           malr.defaults[0] / 10,
                                           malr.defaults[0] * 2,
                                           malr.defaults[0] / 2]}}

trainingErrors = {}
testErrors = {}
goalThreshold = .02

productOfConditions = len(numHiddenLayers) * len(hiddenLayerSize) * len(activationFunctions) * len(trainingMethods) * 5
conditionsExplored = 0

for j in numHiddenLayers:
    for i in hiddenLayerSize:
        # All topologies with no hidden layer are equivalent, so we only need to consider one case
        if j == 0 and i == hiddenLayerSize[0] or j > 0:
            layerNums = []
            if j != 0:
                # the number of nodes in each layer is constant (j) and the number of layers is determined by i
                layerNums = [j] * i
            # there is always an output layer, the number of nodes of which is determined by the number of cols in target
            layerNums.append(numTarget)
            # print "Current Topology:", layerNums
            for aKey in activationFunctions:
                activationFunction = [activationFunctions[aKey]] * len(layerNums)
                # build net, using the current sort of activation function
                net = nl.net.newff(netMinMax, layerNums, transf=activationFunction)
                #net = nl.net.newff(netMinMax, layerNums)
                for tKey in trainingMethods:
                    # Set the training method
                    net.trainf = trainingMethods[tKey]
                    # Iterate over the set of parameter-values for this training method
                    curParamValues = trainingMethodParameters[tKey]['lr']
                    # TODO: introduce other parameters
                    for param in curParamValues:
                        if tKey == 'gradientDescent':
                            # this returns the entire history of errors vs. epochs
                            trainingError = net.train(trainFeatures, trainTarget, epochs=10, show=0,
                                                      goal=goalThreshold, lr=param)
                        elif tKey == 'momentum':
                            # this returns the entire history of errors vs. epochs
                            trainingError = net.train(trainFeatures, trainTarget, epochs=10, show=0,
                                                      goal=goalThreshold, lr=param)
                        elif tKey == 'adaptiveLearningRate':
                            # this returns the entire history of errors vs. epochs
                            trainingError = net.train(trainFeatures, trainTarget, epochs=10, show=0,
                                                      goal=goalThreshold, lr=param)
                        elif tKey == 'm+a':
                            # this returns the entire history of errors vs. epochs
                            trainingError = net.train(trainFeatures, trainTarget, epochs=10, show=0,
                                                      goal=goalThreshold, lr=param)
                        # however, we are only interested in the final error
                        # this value is the SSE, we need to normalize w.r.t. # of training instances
                        trainingErrors[(j, i, aKey, tKey, param)] = trainingError[-1]/numTrainingInstances
                        # run the trained network on the test set
                        testSim = net.sim(testFeatures)
                        # likewise for the test error
                        testError = nl.net.error.SSE.__call__(net.errorf, testSim - testTarget)/numTestInstances
                        testErrors[(j, i, aKey, tKey, param)] = testError
                        conditionsExplored = conditionsExplored + 1
                        progressPercentage = int(floor(Fraction(conditionsExplored, productOfConditions).__float__()*100))
                        sys.stdout.write('\r')
                        # print progress bar
                        sys.stdout.write("[%-20s] %d%%" % ('='*int(progressPercentage/5),progressPercentage))
                        sys.stdout.flush()
print '\n'
print trainingErrors
print testErrors