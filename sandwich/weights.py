import numpy as np
import theano
import theano.tensor as T

__all__ = []

def product(nos):
    return reduce(lambda a, b: a*b, nos)

def make_weights(shape, activation, seed=None, dtype=theano.config.floatX):
    """
    Make weights with values which are known to
    help converge faster based on the activation.

    Supported functions:
    -------------------

    :type shape: tuple
    :type activation:   theano.tensor
                        Check list of supported functions.
    # TODO: complete doc
    :param shape:       shape of the weights
    :param activation:  activation function

    :return:    numpy.ndarray
                The initialized weights.
    """
    ndim = len(shape)

    if ndim == 2:
        w_bound = np.sqrt(6/(shape[0]+shape[1]))
    else:
        w_bound = 1/reduce(lambda a, b: a*b, shape)

    rng = np.random.RandomState(seed=seed)
    W =  np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=shape), dtype=dtype)

    if activation == T.nnet.sigmoid:
        W *= 4
    return W