import numpy as np
import minst
import sandwich as sw
import theano

theano.config.exception_verbosity = 'high'

l1 = sw.layers.InputLayer(shape=(1, 28, 28))
l2 = sw.layers.Flatten(l1)
l3 = sw.layers.FullConn(l2, n_out=500, activation=sw.activations.sigmoid)
l4 = sw.layers.FullConn(l3, n_out=10, activation=sw.activations.sigmoid)

layers = [l1, l2, l3, l4]

nnet = sw.NNet(layers)

# Load the train data
train_images = np.asarray(minst.get_train_images(), dtype='float32').reshape(60000, 1, 28, 28)

train_images[train_images<=128] = 0
train_images[train_images>128] = 1


img = train_images[0, 0]
for i in xrange(28):
    print "".join(map(str, map(int, img[i])))


train_labels = minst.get_train_labels()
target = np.zeros(shape=train_labels.shape + (10, ), dtype='float32')
print target.shape
for i, label in enumerate(train_labels):
    target[i, label] = 1

print train_images.shape, train_labels.shape, target.shape

cost = sw.costs.binary_crossentropy(nnet)
updates = sw.updates.rmsprop(nnet, cost)

nnet.train(train_images, target, cost=cost, updates=updates, \
           stop_when = lambda epoch_no, error: -0.05 < error < 0.05 )

nnet.dump('fullconn.model')

print 'Dumped !'