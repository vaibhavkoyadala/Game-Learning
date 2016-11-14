from itertools import cycle
def to_mini_batches(x, y, n_mini_batches=1):
    mini_batch_size = x.shape[0] / n_mini_batches
    get_batch = lambda array, batch_no: array[mini_batch_size * batch_no: mini_batch_size * (batch_no + 1)]

    iter_no = 0
    for batch_no in cycle(range(n_mini_batches)):
        yield iter_no, (get_batch(x, batch_no), get_batch(y, batch_no))
        iter_no += 1