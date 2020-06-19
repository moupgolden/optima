from dash.dependencies import Input, Output,State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from joblib import load
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import random
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from app2 import app


    
style = {'padding': '1.5em'}

layout = html.Div([
    dcc.Markdown("""
        ## Manage  your Offers with Intelligence
        
    """,style={'fontWeight': 'bold','color':'grey'}),
    

    html.Div(id='prediction-content', style={'fontWeight': 'bold','color':'blue'}),
    html.Div([
        dbc.Button("Probability", id="open")
    ]),
    


    html.Div([
        dcc.Markdown('###### hour'),
        dcc.Slider(
            id='hour',
            min=0,
            max=24,
            step=1,
            value=3,
            marks={n: str(n) for n in range(0, 24, 1)}
        ),
    ], style=style),
        html.Div([
        dcc.Markdown('###### day'),
        dcc.Slider(
            id='day',
            min=1,
            max=31,
            step=1,
            value=3,
            marks={n: str(n) for n in range(1, 31, 1)}
        ),
    ], style=style),
        html.Div([
        dcc.Markdown('###### month'),
        dcc.Slider(
            id='month',
            min=1,
            max=12,
            step=1,
            value=3,
            marks={n: str(n) for n in range(1, 12, 1)}
        ),
    ], style=style),
        html.Div([
        dcc.Markdown('###### year'),
        dcc.Slider(
            id='year',
            min=2019,
            max=2021,
            step=1,
            value=3,
            marks={n: str(n) for n in range(2019, 2021, 1)}
        ),
    ], style=style),
    

        # html.Div([
            
         #   dcc.Graph(id="fig")],style={"max-width": "50px", 'margin':25, 'textAlign': 'center','display': 'inline'}
         #   ),
    
         
        #**************MODEL**********   *************************** 
        
        html.Div(
    [
        
        dbc.Modal(
            [
                dbc.ModalHeader(dcc.Markdown('# Algorithm probability comparison')),
                dbc.ModalBody(

                    #start
                    html.Div([

                        dcc.Graph(id='fig')
                        
                    ])


                 #end
                 ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ml-auto")
                ),
            ],
            id="modal",backdrop="static",size="xl",
        ),
    ]
)
#**************MODEL*****************  **********************************
        
])
'''
@app.callback(Output('output-tab', 'children'),
              [Input('tabs', 'value'),
              Input('randomforest', 'figure'),
              Input('decisiontree', 'figure')
              ])
def render_content(tab,randomforest,decisiontree):
    if tab == '2': return html.Div([
            html.H3('Tab content 2')])'''




@app.callback(
    Output('prediction-content', 'children'),
    [Input('hour', 'value'),
     Input('day', 'value'),
     Input('month', 'value'),
     Input('year', 'value')
     ])

def predict(hour, day, month, year):
    
   #********************************************************************
    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        data.drop([col],axis=1,inplace=True)
        return data
#**********************************************************************
    df = pd.DataFrame(
        columns=['hour',"day","month",'year'],
        data=[[hour, day, month, year]]
    )
    df = encode(df, 'month', 12)
    df = encode(df, 'day', 30)
    df = encode(df, 'hour', 24)
    df = df[['day_sin','day_cos','month_sin','month_cos','year']]
    lg = model[0]
    dtc = model[1]
    rf = model[2]
    knn = model[3]
    svm=model[4]
    xgb=model[5]
    lg1 = lg.predict(df)[0]
    dtc1 = dtc.predict(df)[0]
    rf1 = rf.predict(df)[0]
    knn1 = knn.predict(df)[0]
    svm1 = svm.predict(df)[0]
    xgb1 = xgb.predict(df)[0]
    s= pd.Series([lg1,dtc1,rf1,knn1,svm1,xgb1])
    s=s.map({0: 'ACCEPTE', 1: 'EN ATTENTE',3:'REFFUSE'})
    results =  'LogisticRegression : {} ** DecisionTreeClassifier : {} ** RandomForestClassifier : {} ** KNeighborsClassifier : {} ** svmClassifier : {} ** XGBoostClassifier : {} '.\
                format(s[0], s[1],s[2],s[3],s[4],s[5])


    return results
@app.callback(
    Output('fig', 'figure'),
    [Input('hour', 'value'),
     Input('day', 'value'),
     Input('month', 'value'),
     Input('year', 'value')
     ])
def proba(hour, day, month, year):
    
   #********************************************************************
    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        data.drop([col],axis=1,inplace=True)
        return data
#**********************************************************************
    df = pd.DataFrame(
        columns=['hour',"day","month",'year'],
        data=[[hour, day, month, year]]
    )
    df = encode(df, 'month', 12)
    df = encode(df, 'day', 30)
    df = encode(df, 'hour', 24)
    df = df[['day_sin','day_cos','month_sin','month_cos','year']]

    lg = model[0]
    rf = model[2]
    knn = model[3]
    svm=model[4]
    xg=model[5]
    
    lg1 = lg.predict_proba(df)[0].flatten()
    lg1=lg1.tolist()
    rf1 = rf.predict_proba(df)[0].flatten()
    rf1=rf1.tolist()
    knn1 = knn.predict_proba(df)[0].flatten()
    knn1=knn1.tolist()
    svm1 = svm.predict_proba(df)[0].flatten()
    svm1=svm1.tolist()
    xg1 = xg.predict_proba(df)[0].flatten()
    xg1=xg1.tolist()
#***********************************figure*****************************************
    labels = ['ACCEPTE','EN ATTENTE','REFFUSE']
    
    values_lg = lg1
    values_rf=rf1
    values_knn = knn1
    values_svm=svm1
    values_xg=xg1

    fig = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=values_rf, name="RandomForest"),
              1, 1)
    fig.add_trace(go.Pie(labels=labels, values=values_lg, name="logistic"),
              1, 2)
    fig.add_trace(go.Pie(labels=labels, values=values_knn, name="KNN"),
              2, 1)
    fig.add_trace(go.Pie(labels=labels, values=values_xg, name="xg"),
              2, 2)

# Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.6, hoverinfo="label+percent+name")

    fig.update_layout(
    title_text="prediction",
    margin = dict(t=0, l=0, r=0, b=0),
    
    
    
    # Add annotations in the center of the donut pies.
    
    annotations=[dict(text='RndF', x=0.2, y=0.8, font_size=20, showarrow=False),
                dict(text='LogReg', x=0.83, y=0.8, font_size=20, showarrow=False),
                dict(text='KNN', x=0.16, y=0.2, font_size=20, showarrow=False),
                dict(text='XGBoost', x=0.84, y=0.2, font_size=20, showarrow=False)])

#****************************************figure**********************************************
    return fig

@app.callback(Output(component_id='day', component_property='max'),
              [Input(component_id='month', component_property='value')])
def change(value):
    if value == 2:
        return 28
    else :
        return 31



#**************MODEL CALLBACK**********  
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
#**************MODEL FUNCTION**********  
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
    


model = pickle.load(open('demand_models.pkl','rb'))