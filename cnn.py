import sandwich as sw
import theano
import mnist_analyse

theano.config.exception_verbosity = 'high'

# Build the nnet
l = sw.layers.InputLayer(shape=(1, 28, 28))
l = sw.layers.Conv2D(l, n_features=4, filter_size=(5, 5), seed=47)
l = sw.layers.Pool2D(l, pool_size=(2, 2), activation=sw.activations.relu)
l = sw.layers.Conv2D(l, n_features=6, filter_size=(5, 5), seed=47)
l = sw.layers.Pool2D(l, pool_size=(2, 2), activation=sw.activations.relu)
l = sw.layers.Flatten(l)
l = sw.layers.FullConn(l, n_out=50, seed=47, activation=sw.activations.sigmoid)
l = sw.layers.FullConn(l, 10, seed=47,  activation=sw.activations.sigmoid)

nnet = sw.NNet(l)


cost = sw.costs.binary_crossentropy(nnet) # + sw.costs.L2_reg(nnet)
updates = sw.updates.adadelta(nnet, cost=cost)

# Train on MNIST
last_error = mnist_analyse.train(nnet, cost, updates)

# Test the trained model
mnist_analyse.test(nnet)

# Save the nnet
nnet.dump('cnn.model', extra={"last_error": last_error})