import numpy as np
import pandas as pd 
import plotly.graph_objs as go
from params import colors


def gen_heatmap(df, title, x_title, y_title, h=400):
    """ Generate a heatmap from dataframe

    Args:
        df (DataFrame): pandas dataframe for plotting
        title (str): title of the figure
        x_title (str): name of x-column
        y_title (str): name of y-column
        h (int, optional): height of figure. Defaults to 400.

    Returns:
        figure: generated figure
    """    
    trace = go.Heatmap(z=np.log(df), x=list(df.index), y=list(df.columns), colorscale = 'Viridis', showscale = False)
    layout = go.Layout(
        # title = {'text': title, 'x': 0.2, 'y': 0.9},
        xaxis = {'title': x_title, 'tickfont': {'size': 12}},
        yaxis = {'title': y_title, 'tickfont': {'size': 12}},
        font= {'size': 12, 'color': 'white'},
        height = h,
        paper_bgcolor = colors['panel'],
        margin = {'r':5, 't':5, 'l':5, 'b':5},
        
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def gen_scatter(df, x, y, title, x_title, y_title, h=400):
    """ Generate a scatter figure from dataframe

    Args:
        df (DataFrame): pandas dataframe
        x (str): name of x-column
        y (str): name of y-column
        title (str): title of figure
        x_title (str): name of x-label
        y_title (str): name of y-label
        h (int, optional): [description]. Defaults to 400.

    Returns:
        figure: generated figure
    """    
    trace = go.Scatter(
        x = df[x],
        y = df[y],
        marker = {
            'symbol': 'circle',
            'size': 10,
            'line': {
                'width': 1
            }
        }
    )
    layout = go.Layout(
        # title = {'text': title, 'x': 0.2, 'y': 0.9},
        xaxis = {'title': x_title, 'tickfont': {'size': 12}},
        yaxis = {'title': y_title, 'tickfont': {'size': 12}},
        font= {'size': 12, 'color': 'white'},
        height = h,
        margin = {'r':5, 't':5, 'l':5, 'b':5},
        paper_bgcolor = colors['panel'],

    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def gen_hist(df, x_col, title, x_title, y_title, n=50, h=400):
    """ generate a histogram upon dataframe

    Args:
        df (DataFrame): pandas dataframe
        x (str): name of column
        title (str): title of figure
        x_title (str): name of x-label
        y_title (str): name of y-label
        n (int, optional): number of bins. Defaults to 50.
        h (int, optional): height of figure. Defaults to 400.

    Returns:
        figure: generated figure
    """    
    trace = go.Histogram(
        x = df[x_col],
        nbinsx=n,
    )
    layout = go.Layout(
        xaxis = dict(title=x_title),
        yaxis = dict(title=y_title),
        font=dict(size=10),
        height = h,
        margin = {'r':5, 't':5, 'l':5, 'b':5},
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def gen_scatter_comparison(x1, y1, x2, y2, title, x_title, y_title, 
                           mode1='markers', mode2='lines',
                           label1='label1', label2='label2', h=400):
    """ generate a scatter figure for comparison from two pairs of x- and y-columns

    Args:
        x1 (list): list of x1 values
        y1 (list): list of y1 values
        x2 (list): list of x2 values
        y2 (list): list of y2 values
        title (str): title of figure
        x_title (str): name of x-label
        y_title (str): name of y-label
        mode1 (str, optional): plotting mode for x1-y1 Defaults to 'markers'.
        mode2 (str, optional): plotting mode for x2-y2 Defaults to 'lines'.
        label1 (str, optional): label for x1-y1 Defaults to 'label1'.
        label2 (str, optional): label for x2-y2 Defaults to 'label2'.
        h (int, optional): height of figure. Defaults to 400.

    Returns:
        figure: generated figure
    """    
    x1 = x1.flatten()
    x2 = x2.flatten()
    y1 = y1.flatten()
    y2 = y2.flatten()
    trace1 = go.Scatter(
        x = x1,
        y = y1,
        mode = mode1,
        marker = {
            'symbol': 'circle',
            'size': 15
        },
        name = label1
    )
    trace2 = go.Scatter(
        x = x2,
        y = y2,
        mode = mode2,
        name = label2
    )
    layout = go.Layout(
        title = {'text': title, 'x': 0.5, 'y': 0.95, 'font': {'color': 'black'}},
        xaxis = {'title': x_title, 'tickfont': {'size': 12}},
        yaxis = {'title': y_title, 'tickfont': {'size': 12}},
        font= {'size': 12, 'color': 'white'},
        height = h,
        margin = {'r':5, 't':5, 'l':5, 'b':5},
        legend = {'x':0.05, 'y':0.85, 'bgcolor': colors['tab']},
        paper_bgcolor = colors['panel'],
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig
