import base64
import io

import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import numpy as np
import pandas as pd
from dash import Dash
from dash.dependencies import Input, Output, State

import utils.gen_plot as plt
from models import single_param_regression, double_param_regression
from params import colors, metrics


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H1(['Low-T Model', html.Br()],
                        style={'margin': 'auto', 
                               'textAlign': 'center', 
                               'margin-top': '5%'}),
                html.H3([
                    'This is a app for investigating low temperature \
                    properties, e.g., resistivity, heat capacity, etc..',
                    html.Br(), 
                    html.Br(),
                    'Instruction:',
                    html.Br(),
                    html.Ol([
                        html.Li('Upload your data file and select x and y columns.'),
                        html.Li('Check data on the right panel and specify a range for modeling.'),
                        html.Li('Select the range of expoenent and the metric of model performance'),
                    ])
                ],style={'font-size': '18px', 
                         'textAlign': 'left', 
                         'margin-left': '20px',
                         'margin-top': '30px', 
                         'margin-bottom': '10px'}),
            ]),
            html.Div(style={'boader': '0',
                            'width': '95%',
                            'height': '2px', 
                            'background': '#333',
                            'background-image': 'linear-gradient(to right, #333, #ccc, #333'}),
            dcc.Upload(
                id='upload_data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]), style={
                    'width': '90%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': 'auto',
                    'margin-top': '30px',
                    'margin-bottom': '30px'
                    },
                multiple=False
            ),
            html.Div(id='uploaded_data', 
                     children='', 
                     style={'margin': 'auto', 
                            'textAlign': 'center'}),
            html.Div(id='store_data', 
                     children='', 
                     style={'display': 'none'}),
            html.Div(id='store_selected_data', 
                     children='',
                     style={'display': 'none'}),
            html.Div([
                html.H3('Select X and y columns:', 
                        style={'font-size': '18px', 
                               'textAlign': 'left', 
                               'margin-left': '20px', 
                               'margin-top': '30px', 
                               'margin-bottom': '10px'}),
                html.Div([
                    html.Label('X :  ', 
                               style={'width': '10%', 
                                      'display': 'inline-block', }),
                    dcc.Dropdown(id='select_X_col', 
                                 options=[], 
                                 value=[], 
                                 style={'width': '85%', 
                                        'display': 'inline-block'}),
                ], style={'width': '100%', 
                          'margin-top': '10px', 
                          'margin-bottom': '10px'}),
                html.Div([
                    html.Label('y :  ', 
                               style={'width': '10%', 
                                      'display': 'inline-block', }),
                    dcc.Dropdown(id='select_y_col', 
                                 options=[], 
                                 value=[], 
                                 style={'width': '85%', 
                                        'display': 'inline-block', }),
                ], style={'width': '100%', 
                          'margin-top': '10px', 
                          'margin-bottom': '30px'}),
            ], style={'margin': 'auto', 
                      'textAlign': 'center'}),
            html.Div([
                html.H3('Select data range:', 
                        style={'font-size': '18px', 
                               'textAlign': 'left', 
                               'margin-left': '20px', 
                               'margin-top': '30px', 
                               'margin-bottom': '10px'}),
                html.Div([
                    dcc.RangeSlider(
                        id='select_data_range',
                        min=1,
                        max=2,
                        step=1,
                    ),
                ], style={'width': '80%', 
                          'margin': 'auto'}),
                html.Div(id='data_range_indicator',
                         style={'margin-bottom': '20px'}),
            ], style={'margin': 'auto', 
                      'textAlign': 'center'}),
            html.Div(style={'boader': '0', 
                            'width': '95%', 
                            'height': '2px', 
                            'background': '#333',
                            'background-image': 'linear-gradient(to right, #333, #ccc, #333'}),

            html.Div([
                html.H3('Select exponent range:', 
                        style={'font-size': '18px', 
                               'textAlign': 'left', 
                               'margin-left': '20px', 
                               'margin-top': '30px', 
                               'margin-bottom': '10px'}),
                html.Div([
                    dcc.RangeSlider(
                        id='select_pow_range',
                        min=0.5,
                        max=5,
                        step=0.1,
                        value=[1.0, 2.5],
                        marks={i: str(i) for i in range(1, 6)},
                    ),
                ], style={'width': '70%', 
                          'margin': 'auto', 
                          'margin-bottom': '20px'}),
                html.Div(id='pow_range_indicator', 
                         style={'margin-bottom': '20px'}),
                html.Div(style={'boader': '0', 
                                'width': '95%', 
                                'height': '2px', 
                                'background': '#333',
                                'background-image': 'linear-gradient(to right, #333, #ccc, #333'}),
                html.H3('Select metric for estimation:', 
                        style={'font-size': '18px', 
                               'textAlign': 'left', 
                               'margin-left': '20px', 
                               'margin-top': '30px', 
                               'margin-bottom': '10px'}),
                html.Div([
                    dcc.RadioItems(
                        id='select_metric',
                        options=[
                            {'label': 'Mean Absolute Error (MAE)',
                             'value': 'MAE'},
                            {'label': 'Mean Squared Error (MSE)',
                             'value': 'MSE'},
                            {'label': 'R2', 'value': 'R2'}
                        ],
                        value='MSE',
                        style={'textAlign': 'left'},
                    ),
                ], style={'width': '80%', 
                          'margin': 'auto', 
                          'margin-top': '20px', 
                          'margin-bottom': '20px'}),
                html.Div(style={'boader': '0', 
                                'width': '95%', 
                                'height': '2px', 
                                'background': '#333',
                                'background-image': 'linear-gradient(to right, #333, #ccc, #333'}),
                html.H3('Select test size:', 
                        style={'font-size': '18px', 
                               'textAlign': 'left', 
                               'margin-left': '20px', 
                               'margin-top': '30px', 
                               'margin-bottom': '10px'}),
                html.Div([
                    dcc.Slider(
                        id='select_test_size',
                        min=0.1,
                        max=0.6,
                        step=0.05,
                        value=0.3,
                        marks={i / 10: {'label': str(i * 10) + '%'}
                               for i in range(1, 7)}
                    )
                ], style={'width': '80%', 
                          'margin': 'auto'}),
            ], style={'margin': 'auto', 
                      'textAlign': 'center'}),
        ], style={'width': '24%',
                  'height': '98%',
                  'vertical-align': 'top',
                  'display': 'inline-block',
                  'overflow': 'auto',
                  'font-size': '16px',
                  'margin-left': '30px',
                  'margin-right': '10px',
                  'margin-top': '20px',
                  'margin-bottom': '0px',
                  'background-color': colors['panel'],
                  'border-radius': '10px'}),

        html.Div([
            dcc.Tabs([
                dcc.Tab(
                    label='Raw Data',
                    children=[
                        html.Div(id='plot_raw',
                                 style={'margin': 'auto',
                                        'background-color': colors['panel'],
                                        'width': '90%',
                                        'margin-top': '5%'}),
                        html.Div(id='data_stat',
                                 style={'margin': 'auto',
                                        'background-color': colors['panel'],
                                        'width': '80%',
                                        'margin-top': '3%'}),
                        html.Div(id='selected_data_stat',
                                 style={'margin': 'auto',
                                        'background-color': colors['panel'],
                                        'width': '80%',
                                        'margin-top': '3%'}),
                    ],
                    selected_style={'backgroundColor': colors['panel'],
                                    'color': 'white'}),
                dcc.Tab(
                    label='Single Parameter Model',
                    children=[
                        html.Div(id='plot_single_error',
                                 style={'margin': 'auto',
                                        'margin-top': '5%',
                                        'width': '90%',
                                        'background-color': colors['panel'],
                                        'width': '50%',
                                        'height': '100%',
                                        'display': 'inline-block',
                                        'vertical-align': 'top'}),
                        html.Div(id='plot_single_best', 
                                 style={'margin': 'auto', 
                                        'margin-top': '5%', 
                                        'width': '90%',
                                        'background-color': colors['panel'], 
                                        'width': '50%', 
                                        'height': '100%', 
                                        'display': 'inline-block', 
                                        'vertical-align': 'top'}),
                        html.Div(id='single_regression_result', 
                                 style={'margin': 'auto', 
                                        'margin-top': '5%', 
                                        'width': '90%', 
                                        'background-color': colors['panel']}),
                    ],
                    selected_style={'backgroundColor': colors['panel'],
                                    'color': 'white'}),
                dcc.Tab(
                    label='Double Parameter Model', 
                    children=[
                        html.Div(id='plot_double_error',
                                 style={'margin': 'auto', 
                                        'margin-top': '5%', 
                                        'width': '90%', 
                                        'background-color': colors['panel'], 
                                        'width': '50%', 
                                        'height': '100%', 
                                        'display': 'inline-block', 
                                        'vertical-align': 'top'}),
                        html.Div(id='plot_double_best',
                                 style={'margin': 'auto', 
                                        'margin-top': '5%', 
                                        'width': '90%', 
                                        'background-color': colors['panel'], 
                                        'width': '50%', 
                                        'height': '100%', 
                                        'display': 'inline-block', 
                                        'vertical-align': 'top'}),
                        html.Div(id='double_regression_result', 
                                 style={'margin': 'auto', 
                                        'margin-top': '5%', 
                                        'width': '90%', 
                                        'background-color': colors['panel']}),
                    ], 
                    selected_style={'backgroundColor': colors['panel'], 
                                    'color': 'white'})
            ], 
            colors={'border': colors['border'], 
                    'background': colors['tab']}),
        ], style={'width': '70%', 
                  'height': '98%', 
                  'overflow': 'auto',
                  'font-size': '14px', 
                  'vertical-align': 'top',
                  'display': 'inline-block', 
                  'overflow': 'auto',
                  'margin-left': '10px', 
                  'margin-right': '30px',
                  'margin-top': '20px', 
                  'margin-bottom': '0px',
                  'backgroundColor': colors['panel'],
                  'border-radius': '10px'}),
    ], style={'width': '100hh', 
              'height': '96vh'}),
    html.P('hao.zheng(at)colorado.edu 2019', 
           style={'height': '3vh', 
                  'textAlign': 'center'})
], style={'width': '100hh', 
          'height': '100vh', 
          'background-color': colors['background'], 
          'color': colors['font']})


@app.callback([Output('plot_single_error', 'children'),
               Output('plot_single_best', 'children'),
               Output('single_regression_result', 'children')],
              [Input('store_selected_data', 'children'),
               Input('select_X_col', 'value'),
               Input('select_y_col', 'value'),
               Input('select_metric', 'value'),
               Input('select_pow_range', 'value'),
               Input('select_test_size', 'value')])
def perform_single_regression(df_json, X_col, y_col, metric, pow_range, test_size):
    if all([df_json, X_col, y_col, metric, pow_range]):
        df = pd.read_json(str(df_json))
        X = df[X_col].values.reshape(-1, 1)
        y = df[y_col].values.reshape(-1, 1)
        
        pow_range = [i for i in np.arange(pow_range[0], pow_range[1]+0.1, 0.1)]
        result = single_param_regression(X, y, pow_range, test_size=test_size, metric=metric)
        
        best_p, best_model, best_error = result.get('best_model', None)
        errors_df = result.get('errors', None)
        
        best_prediction = best_model.predict(X ** best_p)
        
        single_error_fig = dcc.Graph(
            figure=plt.gen_scatter(errors_df, 'Exponent', 'Error', f'{metric} vs Power', 'Power', metric, h=600
        ))
        single_best_fig = dcc.Graph(
            figure=plt.gen_scatter_comparison(
                X, y, X, best_prediction, 
                f'Best Model: y = {round(best_model.intercept_[0], 3)} + {round(best_model.coef_[0][0], 3)}\
                     * x ^{round(best_p, 1)}', 
                X_col, y_col, label1='Raw', label2='Prediction', h=600
            )
        )
        model_message = f'Best model: y = {round(best_model.intercept_[0], 3)} + {round(best_model.coef_[0][0], 3)}\
             * x ^{round(best_p, 1)}'
        error_message = f'Model performance: {metric} = {round(best_error, 4)}'
        
        return single_error_fig, single_best_fig, html.H2([model_message, html.Br(), html.Br(), error_message])
    else:
        return [], [], ''

@app.callback([Output('plot_double_error', 'children'),
               Output('plot_double_best', 'children'),
               Output('double_regression_result', 'children')],
              [Input('store_selected_data', 'children'),
               Input('select_X_col', 'value'),
               Input('select_y_col', 'value'),
               Input('select_metric', 'value'),
               Input('select_pow_range', 'value'),
               Input('select_test_size', 'value')])
def perform_double_regression(df_json, X_col, y_col, metric, pow_range, test_size):
    if all([df_json, X_col, y_col, metric, pow_range]):
        df = pd.read_json(str(df_json))
        X = df[X_col].values.reshape(-1, 1)
        y = df[y_col].values.reshape(-1, 1)
        
        pow_range = [i for i in np.arange(pow_range[0], pow_range[1]+0.1, 0.1)]
        result= double_param_regression(X, y, pow_range, test_size=test_size, metric=metric)
        
        best_p1, best_p2, best_model, best_error = result.get('best_model', None)
        errors_df = result.get('errors', None)
        
        best_prediction = best_model.predict(pd.DataFrame([X.flatten() ** best_p1, X.flatten() ** best_p2]).T)
        
        hmap = errors_df.pivot_table(index='P2', columns='P1', values='Error')
        double_error_fig = dcc.Graph(figure = plt.gen_heatmap(hmap, 'Error', 'P1', 'P2', h=600))
        double_best_fig = dcc.Graph(
            figure = plt.gen_scatter_comparison(
                X, y, X, best_prediction, 
                f'Best Model: y = {round(best_model.intercept_[0], 3)} + {round(best_model.coef_[0][0], 3)} \
                    * x ^{round(best_p1, 1)} + {round(best_model.coef_[0][1], 3)} * x ^{round(best_p2, 1)}',
                X_col, y_col, label1='Raw', label2='Prediction', h=600
            )
        )
        model_message = f'Best model: y = {round(best_model.intercept_[0], 3)} + {round(best_model.coef_[0][0], 3)}\
             * x ^{round(best_p1, 1)} + {round(best_model.coef_[0][1], 3)} * x ^{round(best_p2, 2)}'
        error_message = f'Model performance: {metric} = {round(best_error, 4)}'
        
        return double_error_fig, double_best_fig, html.H2([model_message, html.Br(), html.Br(), error_message])
    else:
        return [], [], ''
    

@app.callback(Output('pow_range_indicator', 'children'),
              [Input('select_pow_range', 'value')])
def update_pow_range_indicator(value):
    return f'Selected range: {value[0]} - {value[1]}'
    

@app.callback([Output('store_selected_data', 'children'),
               Output('data_range_indicator', 'children')],
              [Input('store_data', 'children'),
               Input('select_data_range', 'value')])
def update_selected_data(df_json, value):
    if df_json and value:
        df = pd.read_json(str(df_json))
        df_selected = df.iloc[value[0]-1: value[1]]
        message = f'Total: {df.shape[0]} rows. Selected: {value[0] - 1} - {value[1] - 1}'
        return df_selected.to_json(), message
    else:
        return '', ''


@app.callback([Output('store_data', 'children'),
               Output('select_X_col', 'options'),
               Output('select_y_col', 'options'),
               Output('select_data_range', 'max'),
               Output('uploaded_data', 'children')],
              [Input('upload_data', 'contents')],
              [State('upload_data', 'filename')])
def input_data(contents, fname):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        columns = [{'label': col, 'value': col} for col in list(df.columns)]
        message = f'{fname} loaded successfully.'
        return df.to_json(), columns, columns, df.shape[0], message
    else:
        return '', [], [], 1, ''
    
@app.callback(Output('plot_raw', 'children'),
              [Input('store_data', 'children'),
               Input('store_selected_data', 'children'),
               Input('select_X_col', 'value'),
               Input('select_y_col', 'value')])
def plot_data_selection(df_json, df_selected_json, X_col, y_col):
    if all([df_json, X_col, y_col]):
        df = pd.read_json(df_json)
        x1, y1 = df[X_col].values, df[y_col].values
        if df_selected_json:
            df_selected = pd.read_json(df_selected_json)
            x2, y2 = df_selected[X_col].values, df_selected[y_col].values
        else:
            x2, y2 = np.array([]), np.array([])
        fig = plt.gen_scatter_comparison(x1, y1, x2, y2, f'{y_col} vs {X_col}', X_col, y_col, 
                                         mode1='markers', mode2='markers', label1='Raw', label2='Selected', h=600)
        return dcc.Graph(figure = fig, style = {'width': '95%', 
                                                'margin': 'auto', 
                                                'margin-top': '3%'})
    
@app.callback(Output('data_stat', 'children'),
              [Input('store_data', 'children')])
def update_data_stat(df_json):
    if df_json:
        df = pd.read_json(str(df_json))
        data_stat = df.describe().T.reset_index()
        return html.Div([
            html.H3('Raw data statistics: ', 
                    style = {'font-size': '18px', 
                             'textAlign': 'center', 
                             'margin-left': '20px',
                             'margin-top': '30px', 
                             'margin-bottom': '10px'}),
            dt.DataTable(columns = [{'name': i, 'id': i} for i in data_stat.columns],
                         data = data_stat.round(3).to_dict('records'),
                         style_as_list_view = True,
                         style_cell = {'backgroundColor': colors['tab']},
                         style_header = {'backgroundColor': colors['background'],
                                         'fontWeight': 'bold' })
        ])
    


@app.callback(Output('selected_data_stat', 'children'),
              [Input('store_selected_data', 'children')])
def update_selected_data_stat(df_selected_json):
    if df_selected_json:
        df_selected = pd.read_json(str(df_selected_json))
        selected_data_stat = df_selected.describe().T.reset_index()
        return html.Div([
            html.H3('Statistics of selected data: ', 
                    style = {'font-size': '18px', 
                             'textAlign': 'center', 
                             'margin-left': '20px',
                             'margin-top': '30px', 
                             'margin-bottom': '10px'}),
            dt.DataTable(columns = [{'name': i, 'id': i} for i in selected_data_stat.columns],
                         data = selected_data_stat.round(3).to_dict('records'),
                         style_as_list_view = True,
                         style_cell = {'backgroundColor': colors['tab'],},
                         style_header = {'backgroundColor': colors['background'],
                                         'fontWeight': 'bold'})
        ])

if __name__ == '__main__':
    app.run_server(
        host='127.0.0.1',
        port=5000,
        debug=True
    )
