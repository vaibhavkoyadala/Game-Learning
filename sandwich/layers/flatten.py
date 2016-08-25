import base

__all__ = ["Flatten"]

class Flatten(base.Base):
    """
        Represents a Flattening layer.

        Extra Attributes:
        ----------------
        """
    def __init__(self, input):


        super(Flatten, self).__init__(input)
        self.out = self.input.out.flatten(2)
        self.shape = (reduce(lambda a, b: a*b, self.input.shape), )
        self.ndim = 1
        self.params = tuple()

    def __str__(self):
        return '<Flatten layer-{}>'.format(self.layer_no)