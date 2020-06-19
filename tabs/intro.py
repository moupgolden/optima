from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from joblib import load
import numpy as np
import pandas as pd
import calendar
from app2 import app
import plotly.express as px
layout = html.Div([
    dcc.Markdown('# Welcome to Optima '),
    html.Div([

                        dcc.Graph(id='map'),
                        html.Div(id='inter_fig',style={'display':"none"})
                        
                    ])
])
@app.callback(
    Output('map', 'figure'),
    [Input('inter_fig', 'children')
     ])
def fig(inter_fig):

    df = pd.read_csv('D://plt/distance.csv')
    df['lat_pick'] = df['lat_pick'].astype(float)
    df['long_pick'] = df['long_pick'].astype(float)
    df['status']=df['status'].map({0: 'accepted', 1: 'declined',3:"waiting"})
    df['month'] = pd.DatetimeIndex(df['date_time']).month
    df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])
    dfg=df.groupby(['month','lat_dest','long_dest','new_dest_addr','status'])["status"].count().reset_index(name="count")
    
    px.set_mapbox_access_token('pk.eyJ1IjoibW91cCIsImEiOiJja2JmMmpsdmIwcmttMnRwbW8ycXBqZXJjIn0.vOd1OfiuCoMwd6v7zkHTtw')
    #dff = px.data.carshare()
    fig = px.scatter_mapbox(dfg,lat="lat_dest", lon="long_dest",color='status',size='count',hover_name='new_dest_addr',
                    size_max=30, zoom=5,text='new_dest_addr')
    return fig
        