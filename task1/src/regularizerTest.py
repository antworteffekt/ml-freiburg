__author__ = 'Max'
"""
Created on Monday Jun 16 13:15:21 2014

@author: Max
"""

import neurolab as nl
import numpy as np
#import pylab as pl

# set of x values
x = np.linspace(-7,7,20)
# a non-linear set of y values
y = np.square(x) + 5
#y = np.sin(x) * .5
size = len(x)
inp = x.reshape(size,1)
tar = y.reshape(size,1)

# a bunch of neural networks with the same topology but different values for regularization
net1 = nl.net.newff([[-100,100]], [5, 1])
net1.trainf = nl.net.train.train_gd
net2 = net1.copy()
net3 = net1.copy()
net2.regularizer = .5
net3.regularizer = 5

# all networks receive the same training
error1 = net1.train(inp, tar, epochs = 500, show = 0, goal = .02)
error2 = net2.train(inp, tar, epochs = 500, show = 0, goal = .02)
error3 = net3.train(inp, tar, epochs = 500, show = 0, goal = .02)

assert len(net1.layers) == len(net2.layers)
assert len(net2.layers) == len(net3.layers)

weights1 = 0
weights2 = 0
weights3 = 0
# iterate over the layers in all nets, accumulating the weights
for ln, layer in enumerate(net1.layers):
    weights1 += np.sum(net1.layers[ln].np['w']) + np.sum(net1.layers[ln].np['b'])
    weights2 += np.sum(net2.layers[ln].np['w']) + np.sum(net2.layers[ln].np['b'])
    weights3 += np.sum(net3.layers[ln].np['w']) + np.sum(net3.layers[ln].np['b'])

print weights1 ,":", weights2, ":", weights3