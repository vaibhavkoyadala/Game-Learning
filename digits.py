import theano
import theano.tensor as T
import numpy as np

import nnet
import nnet.layers as layers

from itertools import chain

theano.config.exception_verbosity = 'high'

class Digit(object):
    def __init__(self, pixels, label):
        self.pixels = pixels
        self.label = label

    # def make_bitmap(self):
    #     bitmap = Image.new("1", (32, 32))
    #     print self.pixels
    #     print self.label
    #     bitmap.putdata(self.pixels)
    #     #bitmap = bitmap.resize((128, 128))
    #     self.bitmap = bitmap
    #
    # def get_bitmap(self):
    #     if not hasattr(self, 'bitmap'):
    #         self.make_bitmap()
    #     return self.bitmap
    #
    # def show(self):
    #     self.get_bitmap().show(title=str(self.label))

def block(iterable, n, paint):
    """
    Get a list of the first n items.
    :param paint: function to apply for every item.
    """
    return [paint(item) for i, item in zip(xrange(n), iterable)]


def iter_training_data(name):
    """
    Iterate over the training set as Digit object.
    """
    with open(name) as data_file:
        lines = iter(data_file)
        while True:
            image = block(lines, 33, lambda line: line.strip())
            if not image: break
            pixels = [int(pixel) for pixel in chain(*image[:-1])]
            label = int(image[-1])
            yield Digit(pixels, label)


def extract_training_data(name):
    """
    Returns the bitmaps and labels.
    """
    with open(name) as data_file:
        lines = iter(data_file)
        bitmaps = []
        labels = []
        while True:
            image = block(lines, 33, lambda line: line.strip())
            if not image: break
            bitmaps.append([pixel for pixel in chain(*image[:-1])])
            labels.append(int(image[-1]))
        return bitmaps, labels

def train_and_save(model, archive_name):

    y, o = model.y, model.o
    cost = -T.mean(y * T.log(o) + (1 - y) * T.log(1 - o))

    # Get the training set as lists
    bitmaps, labels = extract_training_data("digits.tra")

    # Convert the training set to numpy.ndarray
    x = np.asarray(bitmaps, dtype=theano.config.floatX)
    n_instances = x.shape[0]
    x = x.reshape((n_instances, 1, 32, 32))
    y = np.zeros((len(labels), 10), dtype=theano.config.floatX)
    for i, label in enumerate(labels):
        y[i, label] = 1

    last_cost = model.train(x, y, cost, learning_rate=1, n_epochs=15000)

    o = model.feedforward(x)
    o = np.argmax(o, axis=1)
    print 'Predicted:', o
    print '   Actual:', np.argmax(y, axis=1)
    error = np.count_nonzero(o - np.argmax(y, axis=1))
    print 'Misclassified {} in {}'.format(error, o.shape[0])

    archive = open('archive_name', 'w')
    extra = {'last_cost': last_cost}
    model.dump(archive, extra=extra)


def fullyconn_load_test(archive_name):
    with open('digits.model') as saved:
        meta, extra, saved_model = nnet.NNet.load(saved)
        print 'Meta: ', meta
        print 'Extra:', extra
        W1, b1, W2, b2 = saved_model['W1'], saved_model['b1'], saved_model['W2'], saved_model['b2']

        import nnet.layers as layers

        input_layer = layers.InputLayer(shape=(32 * 32,))
        hidden_layer = layers.FullConn(input_layer, 700, activation=T.nnet.softmax, W=W1, b=b1)
        last_layer = layers.FullConn(hidden_layer, 10, activation=T.nnet.softmax, W=W2, b=b2)

        all_layers = [input_layer, hidden_layer, last_layer]

        model = nnet.NNet('Handwritten digits', layers=all_layers)

        # Get the training set as lists
        bitmaps, labels = extract_training_data("digits.va")

        # Convert the evaluation set to numpy.ndarray
        x = np.asarray(bitmaps, dtype=theano.config.floatX)
        y = np.zeros((len(labels), 10), dtype=theano.config.floatX)
        for i, label in enumerate(labels):
            y[i, label] = 1

        o = model.feedforward(x)
        o = np.argmax(o, axis=1)
        print 'Predicted:', o
        print '   Actual:', np.argmax(y, axis=1)
        error = np.count_nonzero(o - np.argmax(y, axis=1))
        print 'Misclassified {} in {}'.format(error, o.shape[0])

if __name__ == "__main__":
    import nnet.layers as layers

    activation = T.nnet.sigmoid

    input_layer = layers.InputLayer(shape=(1, 32, 32))
    print 'input_layer -shape', input_layer.shape

    conv1_layer = layers.Conv2D( input_layer, n_features=12, filter_size=(4, 4),
                                  activation=activation)
    print 'conv1_layer -shape', conv1_layer.shape

    pool1_layer = layers.Pool2D( conv1_layer, pool_size=(2, 2))
    print 'pool1_layer -shape', pool1_layer.shape

    flatten_layer = layers.Flatten(pool1_layer)
    print 'flatten_layer -shape', flatten_layer.shape

    last_layer = layers.FullConn(flatten_layer, 10, activation=activation)
    print 'last_layer -shape', last_layer.shape

    all_layers = [input_layer, conv1_layer, pool1_layer, flatten_layer, last_layer]

    model = nnet.NNet('Handwritten digits', layers=all_layers)

    train_and_save(model, 'conv_digits.model')
    #fullyconn_load_test(model)

