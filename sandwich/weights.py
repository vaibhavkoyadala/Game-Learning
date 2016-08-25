import numpy as np
import theano
import theano.tensor as T

__all__ = []



def make_weights(w_shape, n_in, n_out, activation, seed=None, dtype=theano.config.floatX):
    """
    Make weights with values which are known to
    help converge faster based on the activation.

    Supported functions:
    -------------------

    :type shape: t      uple
    :type activation:   theano.tensor
                        Check list of supported functions.

    :param w_shape:       shape of the weights
    :param activation:  activation function

    :return:    numpy.ndarray
                The initialized weights.
    """
    # TODO: complete doc



    w_bound = np.sqrt(6.0/(n_in+n_out))

    rng = np.random.RandomState(seed=seed)
    W =  np.asarray(rng.uniform(low=-w_bound, high=w_bound, size=w_shape), dtype=dtype)

    if activation == T.nnet.sigmoid:
        W *= 4

    return W
