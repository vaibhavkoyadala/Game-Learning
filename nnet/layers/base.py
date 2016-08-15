
__all__ = ["Base"]


class Base(object):
    """
    Represents an abstract layer in a neural network.
    All layers must inherit this class.
    
    Attributes:
    out:    Symbolic theano.tensor which gives the output 
            of the layer without the activation.   
    W:      theano.tensor - weights of the layer.
            This is used to compute gradients of the layer.
    b:      theano.tensor - biases of the layer.
            This is used to compute gradients of the layer.
    """
    
    def __init__(self, layer_no, input, activation, W=None, b=None, seed=None):
        """
        :type layer_no:     int
        :type rng:          numpy.random.RandomState
        :type W:            numpy.ndarray
        :type b:            numpy.ndarray
        
        :param layer_no:    The position of the layer in the 
                            neural network. 
                            This will be used to name this layer 
                            and any theano variables.
        
        :param rng:         Random number generator to be used to 
                            populate the initial values of W, b.
                            (if W, b are None)
     
        :param W:           If W is provided, it will be used instead
                            of populating W using rng.
        
        :param b:           If b is provided, it will be used instead
                            of populating b using rng.
        """
        self.layer_no = layer_no
        self.input = input
        
    
    def __str__(self):
        return '<Base layer {}>'.format(self.layer_no)
        
    
        
