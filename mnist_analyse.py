import theano
import mnist
import numpy as np
def train(nnet, cost, updates):
    train_set, valid_set, test_set = mnist.as_numpy()
    train_x, train_labels = train_set
    valid_x, valid_labels = valid_set
    test_x, test_labels = test_set

    train_x = np.asarray(train_x, dtype=theano.config.floatX).reshape(train_x.shape[:1] + nnet.layers[0].shape)
    valid_x = np.asarray(valid_x, dtype=theano.config.floatX).reshape(valid_x.shape[:1] + nnet.layers[0].shape)
    test_x  = np.asarray( test_x, dtype=theano.config.floatX).reshape( test_x.shape[:1] + nnet.layers[0].shape)

    train_y = to_one_hot(train_labels)
    valid_y = to_one_hot(valid_labels)
    test_y = to_one_hot(test_labels)

    costf = theano.function(inputs=[nnet.x, nnet.y],
                            outputs=cost)

    nnet.train( train_x,
                train_y,
                cost=cost,
                updates=updates,
                stop_when=lambda epoch_no, error: epoch_no == 10,
                n_mini_batches=100)
    print "Train cost, valid cost, test cost = {}, {}, {}".format(costf(train_x, train_y),
                                                                  costf(valid_x, valid_y),
                                                                  costf(test_x, test_y))

def test(nnet):
    train_set, valid_set, test_set = mnist.as_numpy()
    test_x, test_labels = test_set
    test_x = np.asarray(test_x, dtype=theano.config.floatX).reshape(test_x.shape[:1] + nnet.layers[0].shape)
    o = nnet.feedforward(test_x)
    predictions = np.argmax(o, axis=1)
    n_wrong = np.sum(predictions != test_labels)
    print "Misclassfied {}/{}".format(n_wrong, test_labels.shape)

def to_one_hot(labels):
    target = np.zeros(shape=(labels.shape[0], 10), dtype=theano.config.floatX)
    target[np.arange(labels.shape[0]), labels] = 1
    return target