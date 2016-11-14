import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T

def as_numpy(fname = 'mnist.pkl.gz'):
    with gzip.open(fname, 'rb') as f:
       train_set, valid_set, test_set = cPickle.load(f)
    return train_set, valid_set, test_set

def as_theano_shared(fname = 'mnist.pkl.gz'):
    train_set, valid_set, test_set = as_numpy(fname=fname)

    train_x, train_y = shared_dataset(train_set)
    valid_x, valid_y = shared_dataset(valid_set)
    test_x, test_y = shared_dataset(test_set)

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def shared_dataset(data_xy, dtype=None):
    """
    Function that loads the dataset into shared variables

    Source: http://deeplearning.net/tutorial/gettingstarted.html
    """
    if dtype is None:
        dtype = theano.config.floatX

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')
