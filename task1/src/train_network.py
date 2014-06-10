
import os
import numpy as np
import neurolab as nl
from sklearn import preprocessing

# set working directory
os.chdir("/Users/jl/uni-freiburg/ss2014/machineLearning/ml-freiburg/task1")

# load data
data = np.loadtxt("data/abalone_train.csv", skiprows = 1)

# separate input from repsonse variables and rescale
ncol = data.shape[1]
scaler = preprocessing.MinMaxScaler()
inputs = scaler.fit_transform(data[:, 0:ncol-3])
targets = data[:, ncol-3:ncol]

# create network
trans = [nl.trans.LogSig(), nl.trans.LogSig()]
net = nl.net.newff([[0,1]] * inputs.shape[1], [10, 3], trans)

# train using different algorithms
print "********************\nResilient backpropagation:"
err_rprop = nl.train.train_rprop(net, inputs, targets, epochs=100, show=10)

# print "********************\nBFGS:"
# err_bfgs = nl.train.train_bfgs(net, inputs, targets, epochs=100, show=10)

# print "********************\nNewton-CG:"
# err_cg = nl.train.train_bfgs(net, inputs, targets, epochs=100, show=10)

if __name__ == "__main__":
    pass