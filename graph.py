import functools
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from dnn.network import DeepCNNClassifier
from utils import load_data, strcolor
from constants import *

CLF = None

@functools.cache
def get_data_shape():
    X, X_test, y, y_test = load_data()
    return X.shape[1], np.unique(y).shape[0]

def get_figure(n_layers, n_neurons, scalename, W_in, W, W_out, training_layer=None):
    n_input, n_output = get_data_shape()
    indent_y = 0.05
    indent_x = 0.05

    x_range = [0 - indent_x * (X_STEP * (n_layers + 1) - X_DISTANCE),
               (X_STEP * (n_layers + 1) - X_DISTANCE) + indent_x * (X_STEP * (n_layers + 1) - X_DISTANCE)]
    y_range = [0 - indent_y * (Y_STEP * n_neurons - Y_DISTANCE),
               (Y_STEP * n_neurons - Y_DISTANCE) + indent_y * (Y_STEP * n_neurons - Y_DISTANCE)]
    n_internal_max = max(n_input, n_output)

    # YShift for input and output layer
    if n_internal_max > n_neurons:
        delta = Y_STEP * (n_internal_max - n_neurons) / 2
        y_range[0] -= delta; y_range[1] += delta

    # XShift for input and output layer
    x_range[0] -= X_STEP; x_range[1] += X_STEP

    x_diap = x_range[1] - x_range[0]
    y_diap = y_range[1] - y_range[0]

    if x_diap > y_diap:
        delta = (x_diap - y_diap) / 2
        y_range[0] -= delta
        y_range[1] += delta
    else:
        delta = (y_diap - x_diap) / 2
        x_range[0] -= delta
        x_range[1] += delta

    shapes = [dict(type="circle",
                   xref="x", yref="y",
                   fillcolor="#E74A4A" if (training_layer == i) or (training_layer == i + 1) else "#B1BCE1",
                   x0=X_STEP * i, y0=Y_STEP * j, x1=X_STEP * i + NEURON_SIZE, y1=Y_STEP * j + NEURON_SIZE,
                   line_color="#829FBA"
                   ) for i in range(0, n_layers + 1) for j in range(0, n_neurons)]
    # Add input layer
    delta_input = Y_STEP * (n_input - n_neurons) / 2
    shapes += [dict(type="circle",
                    xref="x", yref="y",
                    fillcolor="#E74A4A" if training_layer == 0 else "#B1BCE1",
                    x0=-X_STEP,
                    y0=Y_STEP * i - delta_input,
                    x1=-X_STEP + NEURON_SIZE,
                    y1=Y_STEP * i + NEURON_SIZE - delta_input,
                    line_color="#829FBA"
                    ) for i in range(0, n_input)]
    # Add output layer
    delta_output = Y_STEP * (n_output - n_neurons) / 2
    shapes += [dict(type="circle",
                    xref="x", yref="y",
                    fillcolor="#E74A4A" if training_layer == n_layers + 1 else "#B1BCE1",
                    x0=X_STEP * (n_layers + 1),
                    y0=Y_STEP * i - delta_output,
                    x1=X_STEP * (n_layers + 1) + NEURON_SIZE,
                    y1=Y_STEP * i + NEURON_SIZE - delta_output,
                    line_color="#829FBA"
                    ) for i in range(0, n_output)]

    x_ar = []
    y_ar = []
    color_ar = []
    for i in range(0, n_layers):
        for j in range(0, n_neurons):
            for k in range(0, n_neurons):
                x_ar.append(X_STEP * i + NEURON_SIZE)
                x_ar.append(X_STEP * (i + 1))
                y_ar.append(Y_STEP * j + NEURON_SIZE / 2)
                y_ar.append(Y_STEP * k + NEURON_SIZE / 2)
                color_ar.append(W[i][n_neurons - 1 - j][n_neurons - 1 - k])
                color_ar.append(W[i][n_neurons - 1 - j][n_neurons - 1 - k])

    for j in range(0, n_input):
        for k in range(0, n_neurons):
            x_ar.append(-X_DISTANCE)
            x_ar.append(0)
            y_ar.append(Y_STEP * j + NEURON_SIZE / 2 - delta_input)
            y_ar.append(Y_STEP * k + NEURON_SIZE / 2)
            color_ar.append(W_in[n_input - 1 - j][n_neurons - 1 - k])
            color_ar.append(W_in[n_input - 1 - j][n_neurons - 1 - k])

    for j in range(0, n_neurons):
        for k in range(0, n_output):
            x_ar.append(X_STEP * (n_layers + 1) - X_DISTANCE)
            x_ar.append(X_STEP * (n_layers + 1))
            y_ar.append(Y_STEP * j + NEURON_SIZE / 2)
            y_ar.append(Y_STEP * k + NEURON_SIZE / 2 - delta_output)
            color_ar.append(W_out[n_neurons - 1 - j][n_output - 1 - k])
            color_ar.append(W_out[n_neurons - 1 - j][n_output - 1 - k])
            color_ar.append(W_out[n_neurons - 1 - j][n_output - 1 - k])

    dots = go.Scatter(
        x=x_ar, y=y_ar,
        mode='markers',
        marker=dict(
            cmin=MIN_RANGE,
            cmax=MAX_RANGE,
            color=color_ar,
            size=0,
            colorbar=dict(
                thickness=30,
                xanchor='left',
                title="Weghts"
            ),
            colorscale=scalename,
            line_width=0),
        hoverinfo='skip'
    )

    colorscale = dots.marker.colorscale

    # Add hidden links
    shapes += [
        dict(type="line", xref="x", yref="y",
             line=dict(color=strcolor(colorscale, W[i][n_neurons - 1 - j][n_neurons - 1 - k])),
             x0=X_STEP * i + NEURON_SIZE, y0=Y_STEP * j + NEURON_SIZE / 2, x1=X_STEP * (i + 1),
             y1=Y_STEP * k + NEURON_SIZE / 2, line_width=LINE_WIDTH)
        for i in range(0, n_layers) for j in range(0, n_neurons) for k in range(0, n_neurons)
    ]
    # Add input links
    shapes += [
        dict(type="line", xref="x", yref="y",
             line=dict(color=strcolor(colorscale, W_in[n_input - 1 - j][n_neurons - 1 - k])),
             x0=-X_DISTANCE, y0=Y_STEP * j + NEURON_SIZE / 2 - delta_input, x1=0, y1=Y_STEP * k + NEURON_SIZE / 2,
             line_width=LINE_WIDTH)
        for j in range(0, n_input) for k in range(0, n_neurons)
    ]
    # Add output links
    shapes += [
        dict(type="line", xref="x", yref="y",
             line=dict(color=strcolor(colorscale, W_out[n_neurons - 1 - j][n_output - 1 - k])),
             x0=X_STEP * (n_layers + 1) - X_DISTANCE, y0=Y_STEP * j + NEURON_SIZE / 2, x1=X_STEP * (n_layers + 1),
             y1=Y_STEP * k + NEURON_SIZE / 2 - delta_output, line_width=LINE_WIDTH)
        for j in range(0, n_neurons) for k in range(0, n_output)
    ]

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=800,
        height=800,
    )
    fig = go.Figure(data=[dots], layout=layout)

    fig.update_xaxes(range=x_range, zeroline=False, showgrid=False, visible=False)
    fig.update_yaxes(range=y_range, zeroline=False, showgrid=False, visible=False)
    fig.update_layout(shapes=shapes)
    return fig

def render_graph(n_layers, n_neurons, scalename):
    if n_layers is None or n_neurons is None:
        raise PreventUpdate
    if scalename is None:
        scalename = DEFAULT_SCALENAME

    n_input, n_output = get_data_shape()

    W_in = np.random.normal(size=(n_input, n_neurons))
    W = [np.random.normal(size=(n_neurons, n_neurons)) for _ in range(n_layers)]
    W_out = np.random.normal(size=(n_neurons, n_output))
    return get_figure(n_layers, n_neurons, scalename, W_in, W, W_out), True


def init_train(n_layers, n_neurons, activation, batch_size,max_opt_iter, max_iter,
               optimizer, lr, l2, method, scalename):
    print(method)
    global CLF
    X, X_test, y, y_test = load_data()
    CLF = DeepCNNClassifier(
        verbose=True,
        n_layers=n_layers,
        n_features=n_neurons,
        batch_size=batch_size,
        max_opt_iter=max_opt_iter,
        max_iter=max_iter,
        activation=activation,
        optimizer=optimizer,
        lr=lr,
        l2=l2,
        method=method
    )
    CLF.fit(X, y, X_test, y_test)
    W_in = CLF._layers[0].W.detach().numpy()
    W = [CLF._layers[i].W.detach().numpy() for i in range(1, len(CLF._layers) - 1)]
    W_out = CLF._layers[-1].W.detach().numpy()
    return get_figure(n_layers, n_neurons, scalename, W_in, W, W_out), False

def update_train(n_layers, n_neurons, scalename):
    global CLF
    stop_training, training_layer = CLF.train()
    W_in = CLF._layers[0].W.detach().numpy()
    W = [CLF._layers[i].W.detach().numpy() for i in range(1, len(CLF._layers) - 1)]
    W_out = CLF._layers[-1].W.detach().numpy()
    return get_figure(n_layers, n_neurons, scalename, W_in, W, W_out, training_layer), stop_training
