from dash.dependencies import Input, Output,State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from joblib import load
import numpy as np
import pandas as pd
import requests
import re
import pickle
import dash_bootstrap_components as dbc
from app2 import app

layout = html.Div([
    dcc.Markdown("""
        ## what is My FUTURE Income ?
       
    """),

    dcc.Graph(id='hour'),
    html.Div(id='inter_hour',style={'display':"none"}),
    dcc.Graph(id='loss_hour'),
    html.Div(id='inter_hour_loss',style={'display':"none"}),

    dcc.Graph(id='week'),
    html.Div(id='inter_week',style={'display':"none"}),
    dcc.Graph(id='loss_week'),
    html.Div(id='inter_week_loss',style={'display':"none"}),
    

    dcc.Graph(id='month'),
    html.Div(id='inter_month',style={'display':"none"}),
    dcc.Graph(id='loss_month'),
    html.Div(id='inter_month_loss',style={'display':"none"})
    

 
])

@app.callback(
    [Output('hour', 'figure'),Output('week', 'figure'),Output('month', 'figure')],
    [Input('inter_hour', 'children')])
def cnn(inter_hour):
    hour = pickle.load(open('pickle/income_per_hour.pkl','rb'))
    week = pickle.load(open('pickle/income_per_week.pkl','rb'))
    month = pickle.load(open('pickle/income_per_month.pkl','rb'))
    return hour,week,month
@app.callback(
    [Output('loss_hour', 'figure'),Output('loss_week', 'figure'),Output('loss_month', 'figure')],
    [Input('inter_hour_loss', 'children')])

def cn(inter):
    hourloss =  pickle.load(open('pickle/income_per_hour_loss.pkl','rb'))
    weekloss =  pickle.load(open('pickle/income_per_week_loss.pkl','rb'))
    monthloss =  pickle.load(open('pickle/income_per_month_loss.pkl','rb'))
    return hourloss,weekloss,monthloss



