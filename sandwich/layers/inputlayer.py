import base
import theano
import theano.tensor as T

__all__ = ["InputLayer"]

class InputLayer(base.Base):
    """
        Represents an input layer to a neural network.

        Extra Attributes:
        ----------------

        """

    def __init__(self, shape):
        """
        :param input:   shared theano variable holding the input for the
        :param shape:   shape of a single input instance
                        e.g.for 32x32 B&W images, shape = (32x32, ) or (1, 32, 32)
                                                          depending on the receiving layer.
                            for 32x32 RGB images, shape = (3, 32, 32)
        """
        super(InputLayer, self).__init__(None)
        self.params = tuple()
        self.shape = shape
        self.ndim = len(shape)
        nd_tensor = T.TensorType(theano.config.floatX,
                             broadcastable=(False, )*(self.ndim+1)) # note the +1
        self.out = nd_tensor()


    def __str__(self):
        return '<Input layer-{}D {}>'.format(self.ndim, self.layer_no)



