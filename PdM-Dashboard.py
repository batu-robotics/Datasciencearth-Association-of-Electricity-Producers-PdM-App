#%% Full-Stack ML Analyses Dashboard
#%% Designed by: SUMERLabs from Sumertech

#%% Importing Front-End Libraries
import plotly
import plotly.offline as pyo
import plotly.graph_objs as pgo
from plotly import tools

from dash import Dash, dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output,State

import pandas as pd
import numpy as np

#%% Step 1: Gathering Analyses Info
corr_old=pd.read_csv('old.csv')

#%% Step 2: App Design
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
                html.H1("RASAT ML Software"),
                html.Div([
                            dcc.Tabs(
                                id="tabs-with-classes-2",
                                value='tab-1',
                                parent_className='custom-tabs',
                                className='custom-tabs-container',
                                children=[
                                    dcc.Tab(
                                        label='Veri Önişleme',
                                        value='tab-1',
                                        className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                    dcc.Tab(
                                        label='Öznitelik Bilgisi',
                                        value='tab-2',
                                        className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                    dcc.Tab(
                                        label='Model Grafikleri',
                                        value='tab-3', className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                    dcc.Tab(
                                        label='Analiz Sonuçları',
                                        value='tab-4',
                                        className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                ]),
                            html.Div(id='tabs-content-classes-2')
                        ]),
                html.H4("Copyright of SUMERLabs by Sumer Technology")
                ])

@app.callback(Output('tabs-content-classes-2', 'children'),
              Input('tabs-with-classes-2', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Veri Ön Analizi'),
            html.Div([])
            ])

    elif tab == 'tab-2':
        return html.Div([
            html.H3('Öznitelik Bilgisi'),
            html.Div([
                dcc.Graph(id='feature-map',
                          style={'width': '60vh', 'height': '60vh'},
                          figure={'data': [
                              pgo.Heatmap(
                                  z=corr_old,
                                  x=corr_old.columns.values,
                                  y=corr_old.columns.values,
                                  colorscale='Jet',
                                  zmin=-1,
                                  zmax=1
                              )],
                              'layout': pgo.Layout(title='Öznitelik Seçimi Öncesi Model Girdileri')}
                          )
            ],style={'border':"5px black solid"})
            ])

    elif tab == 'tab-3':
        return html.Div([
            html.H3('Model Analizleri'),
            html.Div([])
            ])
            
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Model R2 Skorları'),
            html.Div([])
            ])

#%% Main
app.run_server(debug=True)