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
        :type shape:    tuple
        :param shape:   shape of a single input instance
                        e.g.for 32x32 B&W images, shape = (32x32, ) or (32, 32)
                                                          depending on the next layer.
                            for 32x32 RGB images, shape = (3, 32, 32)
        """
        super(InputLayer, self).__init__(0, None)
        self.params = tuple()
        self.shape = shape
        self.ndim = len(shape)
        ndim = self.ndim
        self.out = T.TensorType(theano.config.floatX, broadcastable=(False, )*+(ndim+1))() # note the +1


    def __str__(self):
        return '<Input layer-{}D {}>'.format(self.ndim, self.layer_no)



