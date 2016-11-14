import theano.tensor as T

def binary_crossentropy(nnet):
    o, y = nnet.o, nnet.y
    return T.nnet.binary_crossentropy(o, y).mean()

def categorial_crossentropy(nnet):
    o, y = nnet.o, nnet.y
    return T.nnet.categorical_crossentropy(o, y).mean()

def L1_reg(nnet, weightage=0.01):
    o, y = nnet.o, nnet.y
    L1 = 0
    for layer in nnet:
        if hasattr(layer, 'W'):
            L1 += (layer.W).sum()
    return weightage*L1

def L2_reg(nnet, weightage=0.01):
    o, y = nnet.o, nnet.y
    L2 = 0
    for layer in nnet:
        if hasattr(layer, 'W'):
            L2 += (layer.W**2).sum()
    return weightage*L2


