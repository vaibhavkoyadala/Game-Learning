import theano
import theano.tensor as T
import cPickle
from layers.inputlayer import InputLayer
import numpy as np
from itertools import cycle

class NNet():
    """
    Represents a Neural Network.

    Attributes:
    ----------
    x: Input
    y: Target output
    o: Predicted ouput. (Symbolic theano variable which computes the output.)
    """

    def __init__(self, last_layer):

        layers = []
        while last_layer is not None:
            layers.append(last_layer)
            last_layer = last_layer.input
        layers.reverse()

        assert isinstance(layers[0], InputLayer), "The first layer in the Neural Network must be "\
                                                  "an Input Layer."
        self.layers = layers
        self.x = layers[0].out
        nd_tensor = T.TensorType(theano.config.floatX,
                                 broadcastable=(False,) * (layers[-1].ndim + 1))  # note the +1
        self.y = nd_tensor()
        self.o = layers[-1].out
        self.o_function = theano.function([self.x], self.o)

    def feedforward(self, x):
        return self.o_function(x)

    def train(self, x, y, cost, updates, stop_when, n_mini_batches=1):
        epoch_no, error = 0, None
        backprop = theano.function(inputs=[self.x, self.y], outputs=cost, updates=updates)

        mini_batch_size = x.shape[0]/n_mini_batches
        get_batch = lambda array, batch_no: array[mini_batch_size*batch_no: mini_batch_size*(batch_no+1)]

        for batch_no in cycle(range(n_mini_batches)):
            mini_x, mini_y = get_batch(x, batch_no), get_batch(y, batch_no)
            error = backprop(mini_x, mini_y)
            if batch_no == 0:
                epoch_no += 1
                print "Epoch {} : error = {}".format(epoch_no, error)
            if stop_when(epoch_no, error):
                break


        return error

    def dump(self, fname, extra=None):
        if extra is None:
            extra = {}

        meta = {'n_layers': len(self.layers),
                'layers': str(self)}

        params = { param.name: param.get_value(borrow=True)  for layer in self for param in layer}

        with open(fname, 'wb') as f:
            cPickle.dump(meta, f, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(extra, f, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(params, f, cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            meta = cPickle.load(f)
            extra = cPickle.load(f)
            params = cPickle.load(f)

        return meta, extra, params

    def __iter__(self):
        return iter(self.layers)

    def __str__(self):
        str = ''
        for layer in self.layers:
            str += '{}{} | '.format(layer, layer.shape)

        return str

