import base
import theano
import numpy as np
from theano.tensor.nnet import conv2d
from .. import weights

__all__ = ["Conv2d"]

class Conv(base.Base):
    """
    Represents a Convolution Neural layer.
    Attributes:
    ----------
    n_features: number of features in this layer
    """
    pass

class Conv2d(Conv):
    def __init__(self, layer_no, input, n_features, filter_size, activation, stride=(1, 1), n_input_features=None, W=None, b=None, seed=None):

        self.layer_no = layer_no

        if isinstance(input, Conv):
            self.input = input.out
            # Make sure n_input_features matches the :input or is not provided.
            if n_input_features is not None and n_input_features != input.n_features:
                print   "Warning: Provided n_input_features({}) and n_features({}) of 'input'"\
                        "do not match. Using value of 'input'.".format(n_input_features, input.n_features)
                n_input_features = n_input_features
        else:
            self.input = input

        self.n_features = n_input_features
        self.filter_size = filter_size
        self.activation = activation
        self.stride = stride

        w_shape = (n_features, n_input_features, filter_size[0], filter_size[1])
        if W is None:
            W = weights.make_weights(w_shape, seed=seed, activation=activation)
        else:
            assert W.shape == w_shape,  "Shape of given weights {} does not match the"\
                                        "shape inferred from the given parameters. {}".format(W.shape, w_shape)
        self.W = theano.shared(W, name='W{}'.format(layer_no))

        b_shape = (n_features, )
        if b is None:
            b = np.zeros(b_shape, dtype=theano.config.floatX)
        else:
            assert b.shape == b_shape,  "Shape of given biases {} does not match the "\
                                        "shape inferred from the given parameters {}.".format(b.shape, b_shape)
        self.b = theano.shared(W, name='b{}'.format(layer_no))

        # Build the symbolic expression that computes the convolution
        conv_out = conv2d(self.input, self.W, subsample=stride)

        self.out = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
