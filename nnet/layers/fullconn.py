from base import Base
from ..weights import make_weights
import numpy as np
import theano
import theano.tensor as T

__all__ = ["FullConn"]

class FullConn(Base):
    """
    Represents a fully connected layer.
    
    Attributes:
    input:  theano.tensor - input to the layer.
    out:    Symbolic theano.tensor which gives the output 
            of the layer without the activation.   
    W:      theano.tensor - weights of the layer.
            This is used to compute gradients of the layer.
    b:      theano.tensor - biases of the layer.
            This is used to compute gradients of the layer.
    """
    
    def __init__(self, layer_no, input, n_in, n_out, activation, W=None, b=None, seed=None):
        """
        :type layer_no:     int
        :type input:        Another layer or theano.tensor
        :type n_in:         int
        :type n_out:        int
        :type activation:   theano.tensor
        :type W:            numpy.ndarray
        :type b:            numpy.ndarray
        :type seed:         int or None
        
        :param layer_no:    The position of the layer in the 
                            neural network. 
                            This will be used to name this layer 
                            and any theano variables.

        :param n_in:       number of input connections
        
        :param n_out:      number of output connections
        
        :param W:           If W is provided, it will be used instead
                            of populating W using weights.make_weights.
                            (In which case, :seed will be used)
        :param b:           If b is provided, it will be used instead
                            of populating b using weights.make_weights.
                            (In which case, :seed will be used)
        """
        self.layer_no = layer_no
        if isinstance(input, Base):
            self.input = input.out
        else:
            self.input = input
        self.n_in = n_in
        self.n_out = n_out

        w_shape = (n_in, n_out)
        if W is None:
            W = make_weights(shape=w_shape, activation=activation, seed=seed)
        elif isinstance(W, np.ndarray):
            assert W.shape == w_shape,  "Shape of given W does not match given n_in, n_out. " \
                                        "{} != {}".format(W.shape, w_shape)
        else:
            raise Exception("Unsupported W type {}.".format(type(W)))

        b_shape = (n_out, )
        if b is None:
            b = np.zeros(
                shape=b_shape,
                dtype=theano.config.floatX,
            )
        elif isinstance(b, np.ndarray):
            assert b.shape == b_shape,  "Shape of given b does not match given n_in, n_out. " \
                                        "{} != {}".format(b.shape, b_shape)
        else:
            raise Exception("Unsupported b type {}.".format(type(b)))

        # Uptil now, W, b are of type numpy.ndarray,
        # make them theano's shared variables.
        self.W = theano.shared(W,
                               name='W{}'.format(layer_no),
                               borrow=True)
        self.b = theano.shared(b,
                               name='b{}'.format(layer_no),
                               borrow=True)

        self.out = activation(T.dot(self.input, self.W) + self.b)

    def __str__(self):
        return '<FullConn layer {}>'.format(self.layer_no)
