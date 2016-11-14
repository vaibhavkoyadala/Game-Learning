#!/usr/bin python
import numpy as np
import sandwich as sw
import theano

theano.config.exception_verbosity = 'high'

meta, extra, params = sw.NNet.load('conv500.model')
print 'meta ::', meta
print 'extra ::', extra
print 'params ::', list(params.keys())


W1, b1 =   None, None  # params['W1'], params['b1']
W3, b3 =   None, None  # params['W3'], params['b3']
W6, b6 =   None, None  # params['W6'], params['b6']
W7, b7 =   None, None  # params['W7'], params['b7']


l = sw.layers.InputLayer(shape=(1, 28, 28))
l = sw.layers.Conv2D(l, n_features=4, filter_size=(5, 5), seed=47, W=W1, b=b1)
l = sw.layers.Pool2D(l, pool_size=(2, 2), activation=sw.activations.relu)
l = sw.layers.Conv2D(l, n_features=6, filter_size=(5, 5), seed=47, W=W3, b=b3)
l = sw.layers.Pool2D(l, pool_size=(2, 2), activation=sw.activations.relu)
l = sw.layers.Flatten(l)
l = sw.layers.FullConn(l, n_out=50, seed=47, activation=sw.activations.sigmoid, W=W6, b=b6)
l = sw.layers.FullConn(l, 10, seed=47,  activation=sw.activations.sigmoid, W=W7, b=b7)

nnet = sw.NNet(l)

import mnist_analyse
cost = sw.costs.binary_crossentropy(nnet) #+ sw.costs.L2_reg(nnet)
updates = sw.updates.adadelta(nnet, cost=cost)
mnist_analyse.train(nnet, cost, updates)
mnist_analyse.test(nnet)