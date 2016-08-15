import theano
import theano.tensor as T
import cPickle
from itertools import izip

class NNet():
    """
    Represents a Neural Network.

    Attributes:
    ----------
    name:   name of the nnet.
    layers: list or tuple of the layers.
    o:  theano.tensor which represents the
        computed output of the neural network.
    x: input to the neural network
    """

    def __init__(self, name, x, y, layers):
        """
        :type name: str
        :type layers:   list or tuple of layers

        :param name: name of the Neural Network
        :param layers: layers in the Neural Network
        """
        self.name = name
        self.x = x
        self.layers = layers
        self.y = y

        self.o = layers[-1].out
        self.o_function = theano.function([self.x], self.o)

    def train(self, x, y, cost, learning_rate, n_epochs):
        wrt = []
        for layer in self.layers:
            wrt.append(layer.W)
            wrt.append(layer.b)

        grads = T.grad(cost, wrt=wrt)

        updates = {(param, (param - learning_rate * grad)) for param, grad in izip(wrt, grads)}

        train = theano.function([self.x, self.y],
                                outputs=cost,
                                updates=updates)

        prevcost = float("inf")
        for epoch in xrange(n_epochs):
            cost = train(x, y)
            print '{:>4} | {}'.format(epoch, cost)

            if cost > prevcost:
                print 'Warning: Potential jumping in gradient descent'
                print 'Current cost {} is greater than previous cost {}'.format(cost, prevcost)
                prevcost = cost
        return cost

    def feedforward(self, x):
        return self.o_function(x)

    def dump(self, file, extra=None):
        if extra is None:
            extra = {}

        meta = {'n_layers': len(self.layers)}

        model = {}
        for layer in self.layers:
            W = layer.W
            b = layer.b
            model[W.name] = W.get_value()
            model[b.name] = b.get_value()

        cPickle.dump(meta, file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(extra, file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(model, file, cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        meta = cPickle.load(file)
        extra = cPickle.load(file)
        model = cPickle.load(file)

        return meta, extra, model


