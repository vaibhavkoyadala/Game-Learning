import plotly.graph_objs as go
from plotly.offline.offline import plot

def get_costs(fname):
    with open(fname) as f:
        costs = []
        for line in f:
            words = line.split()
            if len(words) == 3 and words[1] == '|':
                costs.append(float(words[2]))
        return costs

if __name__ == '__main__':
    traces = []
    for fname in ['vanilla.txt', 'momentum.txt', 'nesterov.txt', 'adagrad.txt', 'adadelta.txt', 'rmsprop.txt']:
        y=get_costs(fname)
        x=range(1, len(y))
        traces.append(go.Scatter(name=fname, x = x, y = y))
    layout = go.Layout(yaxis=dict(range=[0.0, 0.5]))
    figure = go.Figure(data=traces, layout=layout)
    plot(figure)