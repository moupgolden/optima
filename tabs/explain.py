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
import plotly.graph_objects as go


region=['Tunis','Manouba','ben arous',
        'ariana','Bizerte','beja', 'jandouba', 'nabeul' , 'zaghouane', 'silana', 'kef' ,
        'kasserine' , 'kairouane' , 'Sousse' , 'monastir', 'Mahdia' , 'Sfax' , 'sidi bouzid' , 'gafsa' ,
        'touzeur', 'kbeli' , 'gabes' , 'mednine' ,'tataouine']
goods= ["Solide","liquides alimentaires (lait, huile)","Pulvérulents agro-alimentaire(farine,cacao)","Pulvérulents BTP (ciment)"]

layout = html.Div([
    dcc.Markdown("""
        ## I wonder When will my Transporter arrive ?
        ## Will he be free to take my offer ?
       
    """,style={'fontWeight': 'bold','color':'grey'}),

    

    html.Div([

    html.Br(),
    html.Div(
    [   

    dbc.Label("Departure",color="warning",style={'fontWeight': 'bold'}),
    dcc.Dropdown(
        id='depart',
        options=[
          
            {'label': i.upper(), 'value': i.lower()} for i in region],
        value=region[0]
    )
    ],style = {'maxWidth': '400px', 'margin': 'auto'}
),

    html.Div([
    dbc.Label('Destination ',color="warning",style={'fontWeight': 'bold'}),
    dcc.Dropdown(
        id='destination',
        options=[
          
            {'label': i.upper(), 'value': i.lower()} for i in region],
        value=region[1]
    )],style = {'maxWidth': '400px', 'margin': 'auto'}),

    html.Div([
    dbc.Label('VehicleLoadType ',color="warning",style={'fontWeight': 'bold'}),
    dcc.Dropdown(
        id='vehicleLoadType',
        options=[
          
            {'label': i.upper(), 'value': i} for i in goods],
            
        value=goods[0]
     )],style = {'maxWidth': '400px', 'margin': 'auto'}),

    html.Div([
    dbc.Label('VehicleLoadWeight ',color="warning",style={'fontWeight': 'bold'}),
    dbc.Input(id='vehicleWeight', placeholder='vehicleWeight', type='text')],style = {'maxWidth': '400px', 'margin': 'auto'}),

    html.Div(id='text', style={'fontWeight': 'bold','color':'blue'}),
    html.Br(),
    dbc.Button('Submit', id='button',outline=True, color="warning", className="mr-1",style={'fontWeight': 'bold'})

#*******************************************************************************************

    
#************************************************************************************  

],style = {'maxWidth': '500px','Height': '200px', '500px' 'margin': 'auto','float':'left','border': '3px'}),
    html.Div([
        html.Div(id='inter_range',style={'display':"none"}),
        dcc.Graph(id='range')



    ],style = {'overflow': 'hidden'})


])

@app.callback(
    Output('text', 'children'),
    [Input('button', 'n_clicks')],
    [State('depart', 'value'),
    State('destination', 'value'),
    State('vehicleLoadType', 'value'),
    State('vehicleWeight', 'value')

    
    ])

def arrival(n_clicks,depart, destination, vehicleLoadType, vehicleWeight):
    if not n_clicks:
        raise PreventUpdate
    
    if depart is not None and depart is not '' and destination is not None and destination is not '' and vehicleLoadType is not None and vehicleLoadType is not '' and vehicleWeight is not None and vehicleWeight is not '' :
        
        def convert(city):
            #******************
            URL = "https://geocode.search.hereapi.com/v1/geocode"
            api_key = 'WYXfOvRiftbUIQZVW-Zpi2lJyS6xFXKiTGUDhEaKoAU' # Acquire from developer.here.com
            #******************
            query = str(city)+','+'tunisia'
            PARAMS = {'apikey':api_key,'q':query} 
            r = requests.get(url = URL, params = PARAMS) 
            results = r.json()
            lat_city = results['items'][0]['position']['lat']
            long_city = results['items'][0]['position']['lng']
            return lat_city,long_city
        
        depart=convert(depart)
        dest=convert(destination)

        latdep=depart[0]
        longdep=depart[1]

        latdest=dest[0]
        longdest=dest[1]

        if vehicleLoadType=="Solide":
            vehicleLoadType="USHazmatClass1"
        elif vehicleLoadType=="liquides alimentaires (lait, huile)":
            vehicleLoadType="USHazmatClass2"
        elif vehicleLoadType=="Pulvérulents agro-alimentaire(farine,cacao)":
            vehicleLoadType= "USHazmatClass4"
        elif vehicleLoadType=='Pulvérulents BTP (ciment)':
             vehicleLoadType= "USHazmatClass4"



        
        vt='vehicleLoadType='+vehicleLoadType
        position=str(latdep)+','+str(longdep)+':'+str(latdest)+','+str(longdest)
        limitedWeight='&vehicleWeight='+str(vehicleWeight)
        
        url2='https://api.tomtom.com/routing/1/calculateRoute/'+position+'/json?'+vt+limitedWeight+'&travelMode=truck&key=Ap6fHSe1VWNZcNenKy99OXURgkd8TfiN'

        response2 = requests.get(url2)
        data2 = response2.json()
        result=data2['routes'][0]['legs'][0]['summary']
        

        import re
        d=result['arrivalTime']
        arrivaltime = re.split('-|T|:|\+',d)
        del arrivaltime[5:8]
       # msg=html.P("Drivers arrival time is{}",arrivaltime,className='text-info')
        return str(result)
@app.callback(
    Output('range', 'figure'),
    [Input('inter_range', 'children')])

def range(inter_range):
    
    api_key = 'PYakxYQm1CJZiKtXsKZLxeT-2WuZPEHpIZnfcr-_uvA' # Acquire from developer.here.com
    link='PYakxYQm1CJZiKtXsKZLxeT-2WuZPEHpIZnfcr-_uvA'

    URL = 'https://isoline.route.ls.hereapi.com/routing/7.2/calculateisoline.json?apiKey='+link+'&mode=fastest;truck;traffic:disabled&destination=geo!34.73324,10.75057&range=3600&rangetype=time'


    response = requests.get(URL)
                            
    data = response.json()

    cmp=data['response']['isoline'][0]['component'][0]
    dt=pd.json_normalize(cmp,'shape')
    dt[['latitude','longtitude']] = dt[0].str.split(',',expand=True)
    dt = dt.drop(0, axis=1)

    lati=dt['latitude'].tolist()
    longi=dt['longtitude'].tolist()

    

    mapbox_access_token = 'pk.eyJ1IjoibW91cCIsImEiOiJja2JmMmpsdmIwcmttMnRwbW8ycXBqZXJjIn0.vOd1OfiuCoMwd6v7zkHTtw'

    fig = go.Figure(go.Scattermapbox(
            lat=lati,
            lon=longi,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=9
            ),
            text=["The coffee bar","Bistro Bohem","Black Cat",
                "Snap","Columbia Heights Coffee","Azi's Cafe",
                "Blind Dog Cafe","Le Caprice","Filter",
                "Peregrine","Tryst","The Coupe",
                "Big Bear Cafe"],
        ))

    fig.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=5,
            center=dict(
                lat=34.80001,
                lon=10.18270
            ),
            pitch=2,
            zoom=8
        ),
    )
    return fig


    
      


            
