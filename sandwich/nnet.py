import theano
import theano.tensor as T
import cPickle
from layers.inputlayer import InputLayer
class NNet():
    """
    Represents a Neural Network.

    Attributes:
    ----------

    """

    def __init__(self, name, layers):
        assert isinstance(layers[0], InputLayer), "The first layer in the Neural Network must be "\
                                                  "an Input Layer."
        self.name = name
        self.layers = layers
        self.x = layers[0].out
        out_ndim = layers[-1].ndim
        self.y = T.TensorType(dtype=theano.config.floatX, broadcastable=(False, )*(out_ndim+1))() # note the +1
        self.o = layers[-1].out
        self.o_function = theano.function([self.x], self.o)

    def train(self, x, y, cost, learning_rate, n_epochs, momentum=0):
        assert  0 <= momentum < 1, 'Momentum={} should be between [0, 1).'.format(momentum)

        params = []
        for layer in self.layers:
            params += layer.params

        updates = []
        for param in params:

            grad = T.grad(cost, wrt=param)
            
            # prev_grad = theano.shared(param.get_value()*0, broadcastable=param.broadcastable)
            # updates.append((prev_grad, learning_rate*grad - momentum*prev_grad))
            updates.append((param, param-learning_rate*grad))

        backprop = theano.function([self.x, self.y],
                                   outputs=cost,
                                   updates=updates)

        for epoch in xrange(n_epochs):
            curr_cost = backprop(x, y)
            print '{:>4} | {}'.format(epoch, curr_cost)

        return curr_cost

    def feedforward(self, x):
        return self.o_function(x)

    def dump(self, file, extra=None):
        if extra is None:
            extra = {}

        meta = {'n_layers': len(self.layers)}


        model = {}
        for layer in self.layers:
             for param in layer.params:
                 model[param.name] = param.get_value()


        cPickle.dump(meta, file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(extra, file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(model, file, cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file):
        meta = cPickle.load(file)
        extra = cPickle.load(file)
        model = cPickle.load(file)

        return meta, extra, model


    def __str__(self):
        str = self.name + ' :: '
        for layer in self.layers:
            str += ' |{}{}'.format(layer, layer.shape)

        return str + '|'

