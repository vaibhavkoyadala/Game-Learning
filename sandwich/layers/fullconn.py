import base
from ..weights import make_weights
import numpy as np
import theano
import theano.tensor as T

__all__ = ["FullConn"]


class FullConn(base.Base):
    """
    Represents a fully connected layer.

    Additional Attributes:
    ---------------------
    W:      theano.tensor - weights of the layer.
            This is used to compute gradients of the layer.
    b:      theano.tensor - biases of the layer.
            This is used to compute gradients of the layer.
    """

    def __init__(self, input, n_out, activation=None, W=None, b=None, seed=None):
        """
        :type input:        input to this layer
        :type n_out:        int
        :type activation:   theano.tensor
        :type W:            numpy.ndarray or None
        :type b:            numpy.ndarray or None
        :type seed:         int or None

        :param input:       Input to this layer
        :param n_out:       number of output connections
        :param W:           If W is provided, it will be used instead
                            of populating W using seed.
        :param b:           If b is provided, it will be used instead
                            of populating b using seed.
        :param seed:        seed to use to generate random values for W, b
        """

        assert input.ndim == 1, "Incompatibility: ndim-{} of input-{} is not 1".format(input.ndim, input)

        super(FullConn, self).__init__(input)
        self.shape = (n_out,)
        self.ndim = 1
        self.activation = activation

        n_in, = input.shape
        w_shape = (n_in, n_out)
        if W is None:
            W = make_weights(w_shape, n_in, n_out, activation=activation, seed=seed)
        elif isinstance(W, np.ndarray):
            assert W.shape == w_shape, "Shape of given W does not match given n_in, n_out. " \
                                       "{} != {}".format(W.shape, w_shape)
        else:
            raise Exception("Unsupported W type {}.".format(type(W)))

        b_shape = (n_out,)
        if b is None:
            b = np.zeros(shape=b_shape, dtype=theano.config.floatX)
        elif isinstance(b, np.ndarray):
            assert b.shape == b_shape, "Shape of given b does not match given n_in, n_out. " \
                                       "{} != {}".format(b.shape, b_shape)
        else:
            raise Exception("Unsupported b type {}.".format(type(b)))

        self.W = theano.shared(W,
                               name='W{}'.format(self.layer_no),
                               borrow=True)
        self.b = theano.shared(b,
                               name='b{}'.format(self.layer_no),
                               borrow=True)

        self.params = (self.W, self.b)
        out = T.dot(self.input.out, self.W) + self.b
        self.out = out if activation is None else activation(out)

    def __str__(self):
        return '<FullConn layer {}>'.format(self.layer_no)