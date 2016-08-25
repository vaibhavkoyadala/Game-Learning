import base
import theano
import numpy as np
from theano.tensor.signal import pool
from .. import weights

__all__ = ["Pool2D"]

class Pool2D(base.Base):
    """
        Represents a Pooling layer.
        (Pooling in 2D)

        Extra Attributes:
        ----------------
        pool_size:
        padding:
        stride:

        """
    def __init__(self, input, pool_size, padding=(0, 0), stride=None, mode='max', ignore_border=True, activation=None):

        assert input.ndim >= 3, "Incompatibility: ndim-{} of input-{} is not >= 3".format(input.ndim, input)

        super(Pool2D, self).__init__(input)
        self.stride = pool_size if stride is None else stride
        self.params = tuple()
        self.padding = padding
        self.shape = Pool2D.out_shape(input.shape, pool_size, self.stride, padding, ignore_border)
        self.ndim = len(self.shape)
        self.activation = activation
        out = pool.pool_2d(self.input.out,
                           pool_size,
                           padding=padding,
                           st=stride,
                           ignore_border=ignore_border,
                           mode=mode)
        self.out = out if activation is None else activation(out)

    @staticmethod
    def out_shape(input_shape, pool_size, stride, padding, ignore_border):
        """
        Compute the output shape of a convolution layer.
        :param n_features:      no of features in the convolution layer
        :param filter_size:     (height, width) of filter
        :param stride:          (horizontal, vertical) stride
        :param padding:         (horizontal, vertical) padding
        :return:    None if invalid combination.
                    (n_features, height, width) of output if valid.
        """
        #           input   pool
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

        H, W = input_shape[-2], input_shape[-1]
        h, w = pool_size
        hs, vs = stride
        hp, vp = padding

        vertical_len = (W-w+2*vp) / vs +1
        if not ignore_border and (H - h + 2 * vp) % vs :
            vertical_len += 1

        horizontal_len = (W - w + 2 * hp) / hs + 1
        if not ignore_border and (W - w + 2 * hp) % hs:
            horizontal_len += 1

        return input_shape[:-2] + (vertical_len, horizontal_len)

    def __str__(self):
        return '<Pool2D layer {}>'.format(self.layer_no)
