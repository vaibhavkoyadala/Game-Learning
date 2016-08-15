import numpy as np
import theano

__all__ = []


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
    :param shape: shape of the weights
    :param activation: activation function

    :return:    numpy.ndarray
                The initialized weights.
    """

    rng = np.random.RandomState(seed=seed)
    return np.asarray(rng.uniform(size=shape), dtype=dtype)