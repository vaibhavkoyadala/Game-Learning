import base
import theano
import numpy as np
from theano.tensor.nnet import conv2d
from .. import weights

__all__ = ["Conv2D"]

class Conv2D(base.Base):
    """
        Represents a Convolution Neural layer.
        (Convolution in 2D)

        Extra Attributes:
        ----------------
        n_features:
        filter_size:
        stride:
        padding:
        """
    def __init__(self, input, n_features, filter_size,
                 stride=(1, 1), padding=(0, 0), activation=None, W=None, b=None, seed=None):

        # Minimum ndim of the inut layer must be 3. (n_features, height, width)
        assert input.ndim == 3, "Incompatibility: ndim-{} of input-{} is not 3".format(input.ndim, input)

        super(Conv2D, self).__init__(input)
        self.n_features = n_features
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.shape = Conv2D.out_shape(input.shape, n_features, filter_size, stride, padding)
        self.ndim = len(self.shape)

        # height, width of each filter
        height, width = filter_size
        n_in_features = input.shape[0]
        n_in = n_in_features * height * width
        n_out = n_features * height * width

        # w_shape = (n_features, n_in_features, filter_height, filter_width)
        w_shape = (n_features, n_in_features, height, width)
        if W is None:
            W = weights.make_weights(w_shape, n_in, n_out, seed=seed, activation=activation)
        else:
            assert W.shape == w_shape,  "Shape of given weights {} does not match the"\
                                        "shape inferred from the given parameters. {}".format(W.shape, w_shape)

        b_shape = (n_features, )
        if b is None:
            b = np.zeros(b_shape, dtype=theano.config.floatX)
        else:
            assert b.shape == b_shape,  "Shape of given biases {} does not match the "\
                                        "shape inferred from the given parameters {}.".format(b.shape, b_shape)

        self.W = theano.shared(W, name='W{}'.format(self.layer_no), borrow=True)
        self.b = theano.shared(b, name='b{}'.format(self.layer_no), borrow=True)

        # Build the symbolic expression that computes the convolution
        out = conv2d(input=self.input.out,
                          filters=self.W,
                          filter_shape=w_shape,
                          subsample=stride,
                          input_shape=(None, )+self.input.shape) \
              + self.b.dimshuffle('x', 0, 'x', 'x')

        if activation is not None:
            self.out = activation(out)
        else:
            self.out = out

        self.params = (self.W, self.b)



    @staticmethod
    def out_shape(input_shape, n_features, filter_size, stride, padding):
        """
        Compute the output shape of a convolution layer.
        :param n_features:      no of features in the convolution layer
        :param filter_size:     (height, width) of filter
        :param stride:          (horizontal, vertical) stride
        :param padding:         (horizontal, vertical) padding
        :return:    None if invalid combination.
                    (n_features, height, width) of output if valid.
        """
        #           input   filter
        #   height  H       h
        #   width   W       w
        #
        #   hs      horizontal stride
        #   vs      vertical stride
        #   hp      horizontal padding
        #   vp      vertical padding
        #   h = (H-h+2*hp)/hs +1
        #   v = (V-v+2*vp)/vs +1
        #   Invalid combination when remainder is non-zero.

        H, W = input_shape[1], input_shape[2]
        h, w = filter_size
        hs, vs = stride
        hp, vp = padding
        if (H-h+2*vp) % vs or (W-w+2*hp) % hs:
            raise Exception("Invalid combinations of filter_size, stride, padding for input.")
        return n_features, (H-h+2*vp) / vs +1, (W-w+2*hp) / hs +1

    def __str__(self):
        return '<Con2d layer {}>'.format(self.layer_no)
