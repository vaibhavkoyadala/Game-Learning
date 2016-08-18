import theano
import theano.tensor as T
import numpy as np

import sandwich
import sandwich.layers as layers

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
    # print 'y', theano.pp(y)
    # print 'o', theano.pp(o)

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

    for batch in xrange(6):
        last_cost = model.train(x, y, cost, learning_rate=0.02, n_epochs=5000, momentum=0.5)
        print 'Batch {}'.format(batch)
        o = model.feedforward(x)
        o = np.argmax(o, axis=1)
        error = np.count_nonzero(o - np.argmax(y, axis=1))
        print 'Misclassified {} in {}'.format(error, o.shape[0])
        archive = open(archive_name+str(batch), 'w')
        extra = {'last_cost': last_cost}
        model.dump(archive, extra=extra)
        archive.close()

def test(model):
    # Get the training set as lists
    bitmaps, labels = extract_training_data("digits.va")[:5]

    # Convert the evaluation set to numpy.ndarray
    x = np.asarray(bitmaps, dtype=theano.config.floatX)
    n_instances = x.shape[0]
    x = x.reshape((n_instances, 1, 32, 32))
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
    import sandwich.layers as layers

    activation = T.nnet.sigmoid

    # saved = open('digits.2layerfc012345')
    # meta, extra, saved_model = sandwich.NNet.load(saved)
    # print 'Meta: ', meta
    # print 'Extra:', extra['last_cost']
    # W1, b1, W3, b3 = None, None, None, None # saved_model['W1'], saved_model['b1'], saved_model['W3'], saved_model['b3']
    # W6, b6, W7, b7 = None, None, None, None #saved_model['W6'], saved_model['b6'], saved_model['W7'], saved_model['b7']
    input_layer = layers.InputLayer(shape=(1, 32, 32))

    C1 = layers.Conv2D( input_layer, n_features=4, filter_size=(5, 5),
                                  activation=T.nnet.relu)

    S2 = layers.Pool2D( C1, pool_size=(2, 2))

    C3 = layers.Conv2D(S2, n_features=6, filter_size=(5, 5),
                                activation=T.nnet.relu)

    S4 = layers.Pool2D(C3, pool_size=(2, 2))

    flatten_layer = layers.Flatten(S4)

    fc_layer = layers.FullConn(flatten_layer, 500, activation=activation)

    last_layer = layers.FullConn(fc_layer, 10, activation=activation)

    all_layers = [input_layer, C1, S2, C3, flatten_layer, fc_layer, last_layer]

    model = sandwich.NNet('Handwritten digits', layers=all_layers)

    print str(model)
    train_and_save(model, 'digits.2layerfc')
    # test(model)

