import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from graph import *


scalesnames = px.colors.named_colorscales()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP] + external_stylesheets)
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "28rem",
    "padding": "2rem 1rem 80px 0",
    "background-color": "#f8f9fa",
    "overflow-y": "scroll",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

STYLE_CENTER = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}

sidebar = html.Div(children=[
    html.H4(children='DNN Params'),
    html.Div(children=[
        html.Label('Number of layers'),
        dcc.Input(id="n_layers", value='2', type='number'),
    ], style={'float': 'left', 'display': 'block'}),
    html.Div(children=[
        html.Label('Number of neurons'),
        dcc.Input(id="n_neurons", value='3', type='number'),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block',}),
    html.Div(children=[
        html.Label('Batch size'),
        dcc.Input(id="batch_size", value='5000', type='number'),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block', }),
    html.Div(children=[
        html.Label('Number of optimizations per layer'),
        dcc.Input(id="max_opt_iter", value='100', type='number'),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block', }),
    html.Div(children=[
        html.Label('Number of epochs'),
        dcc.Input(id="max_iter", value='50', type='number'),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block', }),
    html.Div(children=[
        html.Label('Learning rate'),
        dcc.Input(id="lr", value='0.01', type='number'),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block', }),
    html.Div(children=[
        html.Label('L2 Regularization'),
        dcc.Input(id="l2", value='0.001', type='number'),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block', }),
    html.Div(children=[
        html.Label('Аctivation function'),
        dcc.Dropdown(
            id='activation',
            style={'width': '100%'},
            options=[
                {'label': 'Linear', 'value': 'equality'},
                {'label': 'ReLU', 'value': 'relu'},
                {'label': 'Sigmoid', 'value': 'sigmoid'},
                {'label': 'Hyperbolic tangent', 'value': 'hyperbolic_tg'},
                {'label': 'Leaky ReLU', 'value': 'leakyReLU'}
            ],
            value='relu'
        )], style={'float': 'left', 'padding-top': '16px', 'display': 'block', 'width': '83%'}),
    html.Div(children=[
        html.Label('Optimizer'),
        dcc.Dropdown(
            id='optimizer',
            style={'width': '100%'},
            options=[
                {'label': 'RMSprop', 'value': 'rmsprop'},
                {'label': 'Adam', 'value': 'adam'},
                {'label': 'Adamax', 'value': 'adamax'},
                {'label': 'AdamW', 'value': 'adamw'},
                {'label': 'Adadelta', 'value': 'adadelta'}
            ],
            value='adam'
        )], style={'float': 'left', 'padding-top': '16px', 'display': 'block', 'width': '83%'}),
    html.Div(children=[
        html.Label('Integration method'),
        dcc.Dropdown(
            id='method',
            style={'width': '100%'},
            options=[
                {'label': 'Euler', 'value': 'euler'},
                {'label': 'Runge–Kutta', 'value': 'runge'}
            ],
            value='euler'
        )], style={'float': 'left', 'padding-top': '16px', 'display': 'block', 'width': '83%'}),
    html.Div(children=[
        html.Button('TRAIN', id='train-button', n_clicks=0, style={'width': '100%'}),
    ], style={'float': 'left', 'padding-top': '16px', 'display': 'block', 'width': '100%'}),
    html.Div(children=[
        html.Button('STOP', id='stop-button', n_clicks=0, style={'width': '100%'}),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block', 'width': '100%'}),
    html.Div(children=[
        html.Button('CONTINUE', id='continue-button', n_clicks=0, style={'width': '100%'}),
    ], style={'float': 'left', 'padding-top': '4px', 'display': 'block', 'width': '100%'}),
], style=SIDEBAR_STYLE)

content = html.Div(children=[
    html.Div(children=[
        dcc.Dropdown(
            id='colorscale',
            options=[{"value": x, "label": x}
                     for x in scalesnames],
            value='tropic',
            style={'margin': '10px 40px 10px 40px'}
        ),
            ], style=STYLE_CENTER),
    html.Div(children=[
            html.H5(id='graph-title')
        ], style=STYLE_CENTER),
    html.Div(children=[
        dcc.Graph(
                id='graph'
            )
    ], style=STYLE_CENTER),
    dcc.Interval(id='graph-update', interval=500, disabled=True)
], style=CONTENT_STYLE)

app.layout = html.Div(children=[
    sidebar, content
])

@app.callback(
    Output("graph-title", "children"),
    Input("graph-update", "disabled")
)
def title_text(is_disabled):
    return "Continuous Neural Network ({})".format("disabled" if is_disabled else "active")

def continue_train(n_continue, is_disabled, graph):
    if n_continue and is_disabled:
        return graph, False
    return graph, is_disabled


def stop_train(n_stop, graph):
    return graph, True

@app.callback(
    Output("graph", "figure"),
    Output("graph-update", "disabled"),
    Input("n_layers", "value"),
    Input("n_neurons", "value"),
    Input("colorscale", "value"),
    Input("train-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    Input("continue-button", "n_clicks"),
    Input("graph-update", "n_intervals"),
    State("graph", "figure"),
    State("graph-update", "disabled"),
    State("activation", "value"),
    State("batch_size", "value"),
    State("max_opt_iter", "value"),
    State("max_iter", "value"),
    State("optimizer", "value"),
    State("lr", "value"),
    State("l2", "value"),
    State("method", "value")
)
def update_output(n_layers, n_neurons, scalename, n_clicks,
                  n_stop, n_continue, n_intervals, graph, is_disabled, activation,
                  batch_size, max_opt_iter, max_iter, optimizer, lr, l2, method):
    if n_layers:
        n_layers = int(n_layers)
    if n_neurons:
        n_neurons = int(n_neurons)
    if batch_size:
        batch_size = int(batch_size)
    if max_opt_iter:
        max_opt_iter = int(max_opt_iter)
    if max_iter:
        max_iter = int(max_iter)
    if lr:
        lr = float(lr)
    if l2:
        l2 = float(l2)

    ctx = dash.callback_context

    if not ctx.triggered:
        return render_graph(n_layers, n_neurons, scalename)

    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

    PROP_ID_MAPPING = {
        key: {'func': render_graph, 'args': [n_layers, n_neurons, scalename]}
        for key in ['n_layers', 'n_neurons', 'colorscale']
    }
    PROP_ID_MAPPING['train-button'] = {'func': init_train, 'args': [n_layers, n_neurons, activation,
                                                                    batch_size, max_opt_iter, max_iter,
                                                                    optimizer, lr, l2, method, scalename]}
    PROP_ID_MAPPING['graph-update'] = {'func': update_train, 'args': [n_layers, n_neurons, scalename]}
    PROP_ID_MAPPING['stop-button'] = {'func': stop_train, 'args': [n_stop, graph]}
    PROP_ID_MAPPING['continue-button'] = {'func': continue_train, 'args': [n_continue, is_disabled, graph]}

    mapping_info = PROP_ID_MAPPING[prop_id]
    return mapping_info['func'](*mapping_info['args'])

if __name__ == '__main__':
    app.run_server(debug=True)