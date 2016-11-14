import theano
import theano.tensor as T
import numpy as np
from itertools import izip
from collections import OrderedDict

__all__ = ['vanilla',
           'momentum',
           'nesterov_momentum',
           'adagrad',
           'adadelta',
           'rmsprop',
           'adam']


def vanilla(nnet, cost, learning_rate=0.01):
    """

    :param layers:  list/tuple of layers
    :param cost:    symbolic expression for cost
    :return:
    """

    params = tuple()
    for layer in nnet:
        params += layer.params

    grads = theano.grad(cost, wrt=params)

    updates = [(param, param - learning_rate*grad) for param, grad in izip(params, grads)]

    return updates


def momentum(nnet, cost, learning_rate=0.01, decay=0.5):
    """

    :param layers:
    :param cost:
    :param learning_rate:
    :param decay:
    :return:
    """

    params = tuple()
    for layer in nnet:
        params += layer.params

    grads = theano.grad(cost, wrt=params)

    updates = OrderedDict()
    for param, grad in izip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        updates[velocity] =  decay*velocity - learning_rate*grad
        updates[param] = param + updates[velocity]

    return updates


def nesterov_momentum(nnet, cost, learning_rate=0.01, decay=0.9):
    """

        :param layers:
        :param cost:
        :param learning_rate:
        :param decay:
        :return:
    """

    params = tuple()
    for layer in nnet:
        params += layer.params

    grads = theano.grad(cost, wrt=params)

    updates = OrderedDict()
    for param, grad in izip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        # http://arxiv.org/pdf/1212.0901v2.pdf
        # http://cs231n.github.io/neural-networks-3
        velocity_new = decay*velocity - learning_rate * grad
        updates[velocity] = velocity_new
        updates[param] = param + decay * velocity_new - learning_rate * grad

    return updates


def adagrad(nnet, cost, initial_learning_rate=0.01, eps=1e-8):
    params = tuple()
    for layer in nnet:
        params += layer.params

    grads = theano.grad(cost, wrt=params)

    updates = OrderedDict()
    for param, grad in izip(params, grads):
        value = param.get_value(borrow=True)
        accumulated_grad = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        accumulated_grad_new = accumulated_grad + grad**2
        updates[accumulated_grad] = accumulated_grad_new
        updates[param] = param - (initial_learning_rate*grad / T.sqrt(accumulated_grad_new+eps))
        
    return updates


def adadelta(nnet, cost, decay=0.9, eps=1e-8):
    params = tuple()
    for layer in nnet:
        params += layer.params

    grads = theano.grad(cost, wrt=params)

    updates = OrderedDict()
    for param, grad in izip(params, grads):
        value = param.get_value(borrow=True)
        accumulated_update = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        accumulated_grad = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        accumulated_grad_new = decay*accumulated_grad + (1-decay)*(grad ** 2)
        updates[accumulated_grad] = accumulated_grad_new
        delta_update = - T.sqrt(accumulated_update + eps) * grad / T.sqrt(accumulated_grad_new + eps)
        updates[param] = param + delta_update
        updates[accumulated_update] = decay*accumulated_update + (1-decay)*(delta_update**2)


    return updates
    
def rmsprop(nnet, cost, initial_learning_rate=0.01, decay=0.9, eps=1e-8):
    params = tuple()
    for layer in nnet:
        params += layer.params

    grads = theano.grad(cost, wrt=params)

    updates = OrderedDict()
    for param, grad in izip(params, grads):
        value = param.get_value(borrow=True)
        accumulated_grad = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        accumulated_grad_new = decay*accumulated_grad + (1-decay)*(grad ** 2)
        updates[accumulated_grad] = accumulated_grad_new
        updates[param] = param - initial_learning_rate * grad / T.sqrt(accumulated_grad_new + eps)

    return updates

def adam(nnet, cost, initial_learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):

    # -------------------------------------
    # TODO: Incomplete implementation (adam)
    # -------------------------------------

    params = tuple()
    for layer in nnet:
        params += layer.params

    grads = theano.grad(cost, wrt=params)

    updates = OrderedDict()
    for param, grad in izip(params, grads):
        value = param.get_value(borrow=True)
        m = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        v = theano.shared(np.zeros(shape=value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        m_new = beta1*m + (1-beta1)*grad
        v_new = beta2*m + (1-beta2)*(grad**2)

        updates[m] = m_new
        updates[v] = v_new
        updates[param] = param - initial_learning_rate * m_new / (T.sqrt(v_new) + eps)

    return updates
    

