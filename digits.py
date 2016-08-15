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


def init_train_save():
    import nnet.layers as layers

    x, y = T.matrix('x'), T.matrix('y')

    hidden = layers.FullConn(1, x, 32 * 32, 700, activation=T.nnet.softmax)
    last = layers.FullConn(2, hidden, 700, 10, activation=T.nnet.softmax)

    layers = [hidden, last]

    model = nnet.NNet('digits', x, y, layers=[hidden, last])

    o = model.o
    cost = -T.mean(y * T.log(o) + (1 - y) * T.log(1 - o)) + 0.1 * T.mean(model.layers[0].W ** 2) + 0.1 * T.mean(
        model.layers[1].W ** 2)

    # Get the training set as lists
    bitmaps, labels = extract_training_data("digits.tra")

    # Convert the training set to numpy.ndarray
    x = np.asarray(bitmaps, dtype=theano.config.floatX)
    y = np.zeros((len(labels), 10), dtype=theano.config.floatX)
    for i, label in enumerate(labels):
        y[i, label] = 1

    last_cost = model.train(x, y, cost, 0.9, 15000)

    o = model.feedforward(x)
    o = np.argmax(o, axis=1)
    print 'Predicted:', o
    print '   Actual:', np.argmax(y, axis=1)
    error = np.count_nonzero(o - np.argmax(y, axis=1))
    print 'Misclassified {} in {}'.format(error, o.shape[0])

    archive = open('digits.model', 'w')
    extra = {'last_cost': last_cost}
    model.dump(archive, extra=extra)


def load_test():
    with open('digits.model') as saved:
        meta, extra, saved_model = nnet.NNet.load(saved)
        print 'Meta: ', meta
        print 'Extra:', extra
        W1, b1, W2, b2 = saved_model['W1'], saved_model['b1'], saved_model['W2'], saved_model['b2']
        import nnet.layers as layers

        x, y = T.matrix('x'), T.matrix('y')

        hidden = layers.FullConn(1, x, 32 * 32, 700, activation=T.nnet.softmax, W=W1, b=b1)
        last = layers.FullConn(2, hidden, 700, 10, activation=T.nnet.softmax, W=W2, b=b2)

        layers = [hidden, last]

        model = nnet.NNet('digits', x, y, layers=[hidden, last])

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
    # init_train_save()
    load_test()

