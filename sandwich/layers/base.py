
__all__ = ["Base"]


class Base(object):
    """
    Represents an abstract layer in a neural network.
    All layers must inherit this class.
    
    Attributes:
    ----------
    input:  Input to this layer
    out:    Symbolic theano.tensor which gives the output 
            of the layer without the activation.   
    params: tuple of parameters the model is based on
            (include W, b)
    shape:  shape of the output for a single
            input instance (as a vector)
    ndim:   no of dimensions of the output for a single
            input instance (as a vector)

    Note:   Computation is done on a dataset, not on
            a data instance.
            (A dataset is a collection of one or more
             data instances.)
            A dataset has an extra dimension in addition
            to the dimensions of the data instance.
            Layers should hence multiply weights or add biases
            or do any other computation on a dataset, but
            attributes :shape, :ndim refer to a data instance.
            :out, again deals with a dataset.
    """
    
    def __init__(self, layer_no, input):
        """
        :type layer_no:     int
        :type input:        Another layer or theano.tensor

        :param layer_no:    The position of the layer in the
                            neural network.
                            This will be used to name this layer
                            and any theano variables.
        :param input:       Input to this layer
        """
        self.layer_no = layer_no
        self.input = input
        self.out = NotImplemented
        self.params = NotImplemented
        self.shape = NotImplemented
        self.ndim = NotImplemented

    def __str__(self):
        return '<Base layer {}>'.format(self.layer_no)
        
    
        
