class Layer():

    def __init__(self, layer_class, *args, **kwargs):
        self.layer_class = layer_class
        self.args = args
        self.kwargs = kwargs

    def instantiate(self, more_args, more_kwargs):
        args = more_args + self.args
        kwargs = self.kwargs
        if more_kwargs is not None:
            kwargs.update(more_kwargs)

        return self.layer_class(*args, **kwargs)