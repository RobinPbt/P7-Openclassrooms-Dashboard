import torch
import os
import pathlib
import dash
import pickle
import shap
import base64
import datetime
import io
import requests
import json

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_daq as daq

from functions import *
from utils import *

# from pages.model_description import build_model_description
# from pages.predict_existing import build_predict_existing
# from pages.predict_new import build_predict_new

app = Dash(__name__)

# ----------------------------------------------------------------------------------
# Loading heavy files
   
# Loading matrices
# x = pd.read_csv('./datas/real_x_reduced.csv') # Note: this matrix corresponds to 5% of initial dataset for memory consumption issues
# x.index = x['SK_ID_CURR']

# y = pd.read_csv('./datas/real_y_reduced.csv') # Note: this matrix corresponds to 5% of initial dataset for memory consumption issues
# y.index = x['SK_ID_CURR']

# x.drop('SK_ID_CURR', axis=1, inplace=True)
# y.drop('Unnamed: 0', axis=1, inplace=True)

y = get_full_labels()

# Loading model, imputer and preprocessor
load_clf = pickle.load(open('./models/final_model.pkl', 'rb'))
imputer = pickle.load(open('./models/imputer.pkl', 'rb'))
preprocessor = pickle.load(open('./models/preprocessor.pkl', 'rb'))

descriptions_df = pd.read_csv('./datas/var_description.csv')

# Loading explainer and top features
explainer = pickle.load(open('./models/explainer.pkl', 'rb'))
shap_global_top_features = pd.read_csv('./datas/shap_top_feat.csv')

# explainer = shap.LinearExplainer(load_clf, x)
# shap_values_global = explainer.shap_values(x)

# customer_list = list(x.index)
customer_list = find_feature_distribution('SK_ID_CURR')
features_list = get_features_names()

# ----------------------------------------------------------------------------------
# Global Layout

app.layout = html.Div(id="big-app-container", children=[
    
    html.Div(id="banner", className="banner",children=[

        html.Div(id="banner-text", children=[
                html.H5("Openclassrooms - Project 7"),
                html.H6("Customer loan repayment capacity prediction application"),
        ]),

        html.Div(id="banner-logo", children=[
                html.Button(id="learn-more-button", children="LEARN MORE", n_clicks=0),
                html.A(html.Img(id="logo", src=app.get_asset_url("oc-logo.png"))),
        ]),
    ]),

    html.Div(id="app-container", children=[
    
        html.Div(id="tabs", className="tabs", children=[
            dcc.Tabs(id='app-tabs', value='tab-1', className="custom-tabs", children=[
                dcc.Tab(
                    id="tab-1", label='Predict existing customer',value='tab-1', className="custom-tab", selected_className="custom-tab--selected"
                ),
                dcc.Tab(
                    id="tab-2", label='Predict new customer',value='tab-2', className="custom-tab", selected_className="custom-tab--selected"
                ),
                dcc.Tab(
                    id="tab-3", label='Model description',value='tab-3', className="custom-tab", selected_className="custom-tab--selected"
                ),
            ]),
        ]),

        html.Div(id="app-content"),
    ]),
    
    html.Div(id="markdown", className="modal", children=[
        html.Div(id="markdown-container", className="markdown-container", children=[
            
            html.Div(className="close-container",children=[
                html.Button("Close", id="markdown_close", n_clicks=0, className="closeButton"),
            ]),
            
            html.Div(className="markdown-text", children=[
                dcc.Markdown(children=[
                    """
                    ###### Project
                    This purpose of this project is to learn how to build and deploy a dashboard to show results of a model we previously trained 
                    ###### Predict existing customer
                    Display prediction of a customer present in the training set (by selecting customer ID)
                    ###### Predict new customer
                    Display prediction of a new customer present by uploading datas (same format than initial csv files)
                    ###### Model description
                    Display main characteristics of trained model
                    
                    ###### Source dashboard style
                    The dashboard style is inspired by this [Github repository](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-manufacture-spc-dashboard)
                    """
                ]),
            ]),
        ]),
    ]),
])

# ----------------------------------------------------------------------------------
# GLobal callbacks

@app.callback(
    Output('app-content', 'children'),
    Input('app-tabs', 'value')
)

def render_content(selected_tab):
    if selected_tab == 'tab-1':
        return build_predict_existing(customer_list)
    elif selected_tab == 'tab-2':
        return build_predict_new()
    else:
        return build_model_description()
    
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "learn-more-button":
            return {"display": "block"}

    return {"display": "none"}

# ----------------------------------------------------------------------------------
# Global functions and variables

def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)

# General layout for graphs
layout_dict = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "xaxis": dict(showline=False, showgrid=False, zeroline=False),
    "yaxis": dict(showgrid=False, showline=False, zeroline=False),
    "font" : {"color": "darkgray"},
    "autosize": True,
}

# Shap local feature importance fig for tab 1 and 2
def generate_shap_local_feat(explainer, selected_customer, features_list, layout_dict):
    
    # Shap values for features
    shap_values_local = explainer.shap_values(selected_customer)
    
    shap_df_local_abs = pd.Series(abs(shap_values_local), index=features_list)
    top_features = shap_df_local_abs.sort_values(ascending=False)[:10].index

    shap_df_local = pd.Series(shap_values_local, index=features_list)

    non_abs_values = []

    for feat in top_features:
        non_abs_values.append(shap_df_local[feat])
    
    # Generate figure to plot
    plot_df = pd.DataFrame()
    plot_df["Feature name"] = top_features
    plot_df["Shap values"] = non_abs_values
    plot_df = plot_df.sort_values(by="Shap values", key=abs)
    
    colors = []
    for val in plot_df["Shap values"]:
        if val <= 0:
            colors.append('#EC8484')
        else:
            colors.append('#f4d44d')
    
    fig = px.bar(plot_df, x="Shap values", y="Feature name")
    fig.update_layout(layout_dict)
    fig.update_traces(
        marker_color=colors,
        marker_line_color='darkgrey',
        marker_line_width=1, opacity=1
    )
    
    return fig

# ----------------------------------------------------------------------------------
# Layout tab 1 (predict existing)

def build_predict_existing(customer_list):

    return (
        html.Div(id="status-tab-1-container", children=[
            
            html.Div(id="tab-1-container", children=[
            
                # Upper selection panel
                html.Div(id='customer-selection-container', children=[
                    html.P("Customer selection"),
                    html.Br(),
                    dcc.Dropdown(customer_list, value=customer_list[0], id='selection-dropdown'),
                ]),

                # Lower panel with 2 sections
                html.Div(id="tab-1-low-graphs-container",children=[
                    
                    html.Div(id="tab-1-graph-1", children=[
                        
                        generate_section_banner("Prediction"),
                        
                        html.Div(id="tab-1-predict-container",children=[
                            html.P("Status"),
                            html.H2(id="tab-1-display-prediction"),
                        ]),
                        
                        html.Div(id="tab-1-proba-container", children=[
                            html.P("Probability"),
                            daq.Gauge(
                                id="tab-1-proba-gauge",
                                max= 1,
                                min=0,
                                showCurrentValue=True,
                                color="#f4d44d",
                                # color={"gradient":False,"ranges":{"green":[0, 0.5], "red":[0.5, 1]}},
                            ),
                        ]),
                    ]),
                    
                    # html.Div(id="tab-1-graph-2", children=[
                    #     generate_section_banner("Probability"),
                    #     html.H1(id="display-proba"),
                    # ]),
                    
                    html.Div(id="tab-1-graph-2", children=[
                        generate_section_banner("Features importance"),
                        dcc.Graph(id="tab-1-local-feat-importance"),
                    ]),
                ]),
            ]),
        ]),
    )

# ----------------------------------------------------------------------------------
# Callbacks tab 1 (predict existing)

@app.callback(
    Output('tab-1-display-prediction', 'children'),
    Output('tab-1-proba-gauge', 'value'),
    Output('tab-1-local-feat-importance', 'figure'),
    Input('selection-dropdown', 'value'),
)

def compute_predictions_and_features(customer_id):
    
    # We request API with our customer id
    PREDICT_URI = 'https://p7-openclassrooms-api.herokuapp.com/api/predict-existing/'
    response = requests.request(method='GET', url=PREDICT_URI, params={'id_customer' : customer_id})
    dict_response = json.loads(response.content.decode('utf-8'))
    
    probas = dict_response['datas']['proba']
    predictions = dict_response['datas']['prediction']
    
    selected_customer = find_customer_features(customer_id)
    
#     # Predictions
#     probas = load_clf.predict_proba(selected_customer.values.reshape(1, -1))[:,1]
#     predictions = (probas >= 0.5).astype(int)
    
#     probas = float(probas[0])
#     predictions = int(predictions[0])
    
    outputs_pred = np.array(['Accepted','Refused'])
    
    current_prediction = outputs_pred[predictions]
    
    fig = generate_shap_local_feat(explainer, selected_customer, features_list, layout_dict)
    
    return current_prediction, probas, fig

# ----------------------------------------------------------------------------------
# Layout tab 2 (predict new)

def build_predict_new():

    return (
        html.Div(id="status-tab-2-container", children=[
            
            html.Div(id="tab-2-container", children=[
            
                # Upper upload panel
                html.Div(id='upload-customer-container', children=[
                    
                    html.Div(id='upload-buttons', children=[
                        dcc.Upload(id='upload-application', children=[html.Button('application', style=style_dict)], multiple=False),
                        dcc.Upload(id='upload-bureau_balance', children=[html.Button('bureau_bal', style=style_dict)], multiple=False),
                        dcc.Upload(id='upload-bureau', children=[html.Button('bureau', style=style_dict)], multiple=False),
                        dcc.Upload(id='upload-previous_application', children=[html.Button('previous_app', style=style_dict)], multiple=False),
                        dcc.Upload(id='upload-credit_card_balance', children=[html.Button('credit_card', style=style_dict)], multiple=False),
                        dcc.Upload(id='upload-POS_CASH_balance', children=[html.Button('POS_CASH', style=style_dict)], multiple=False),
                        dcc.Upload(id='upload-installments_payments', children=[html.Button('installments', style=style_dict)], multiple=False),
                    ]),
                    
                    html.Div(id='upload-status', children=[
                        dcc.Markdown(id='status-application'),
                        dcc.Markdown(id='status-bureau_balance'),
                        dcc.Markdown(id='status-bureau'),
                        dcc.Markdown(id='status-previous_application'),
                        dcc.Markdown(id='status-credit_card_balance'),
                        dcc.Markdown(id='status-POS_CASH_balance'),
                        dcc.Markdown(id='status-installments_payments'),
                    ]),
                    
                    html.H6(id='output-data-upload')
                ]),
                
                # Lower panel with 2 sections
                html.Div(id="tab-2-low-graphs-container",children=[
                    
                    html.Div(id="tab-2-graph-1", children=[
                        
                        generate_section_banner("Prediction"),
                        
                        html.Div(id="tab-2-predict-container",children=[
                            html.P("Status"),
                            html.H2(id="tab-2-display-prediction"),
                        ]),
                        
                        html.Div(id="tab-2-proba-container", children=[
                            html.P("Probability"),
                            daq.Gauge(
                                id="tab-2-proba-gauge",
                                max= 1,
                                min=0,
                                showCurrentValue=True,
                                color="#f4d44d",
                            ),
                        ]),
                    ]),
                    
                    html.Div(id="tab-2-graph-2", children=[
                        generate_section_banner("Features importance"),
                        dcc.Graph(id="tab-2-local-feat-importance"),
                    ]),
                ]),
            ]),
        ]),
    )

# ----------------------------------------------------------------------------------
# Callbacks tab 2 (predict new)

@app.callback(
    # Uploaded files status output
    Output('status-application', 'children'),
    Output('status-bureau_balance', 'children'),
    Output('status-bureau', 'children'),
    Output('status-previous_application', 'children'),
    Output('status-credit_card_balance', 'children'),
    Output('status-POS_CASH_balance', 'children'),
    Output('status-installments_payments', 'children'),
    
    # Global output to genrate predictions and graph
    Output('tab-2-display-prediction', 'children'),
    Output('tab-2-proba-gauge', 'value'),
    Output('tab-2-local-feat-importance', 'figure'),
    
    # Inputs : all uploaded files
    Input('upload-application', 'contents'),
    Input('upload-bureau_balance', 'contents'),
    Input('upload-bureau', 'contents'),
    Input('upload-previous_application', 'contents'),
    Input('upload-credit_card_balance', 'contents'),
    Input('upload-POS_CASH_balance', 'contents'),
    Input('upload-installments_payments', 'contents'),
)

def update_output(uploaded_application, uploaded_bureau_balance, uploaded_bureau, uploaded_previous_application, uploaded_credit_card_balance, uploaded_POS_CASH_balance, uploaded_installments_payments):
    
    # nb_features expected per file (y.c index)
    shapes = {'application' : 122, 'bureau_balance' : 3, 'bureau' : 17, 'previous_application' : 37, 'credit_card_balance' : 23,
          'POS_CASH_balance' : 8, 'installments_payments' : 8}
    
    # Test if file uploaded with correct shape and save it if ok
    application_df, application_status = test_upload(uploaded_application, shapes['application'])
    bureau_balance_df, bureau_balance_status = test_upload(uploaded_bureau_balance, shapes['bureau_balance'])
    bureau_df, bureau_status = test_upload(uploaded_bureau, shapes['bureau'])
    previous_application_df, previous_application_status = test_upload(uploaded_previous_application, shapes['previous_application'])
    credit_card_balance_df, credit_card_balance_status = test_upload(uploaded_credit_card_balance, shapes['credit_card_balance'])
    POS_CASH_balance_df, POS_CASH_balance_status = test_upload(uploaded_POS_CASH_balance, shapes['POS_CASH_balance'])
    installments_payments_df, installments_payments_status = test_upload(uploaded_installments_payments, shapes['installments_payments'])

    
    # check if all files uploaded correctly
    upload_counter = 0
    
    status_list = [
        application_status, 
        bureau_balance_status, 
        bureau_status, 
        previous_application_status,
        credit_card_balance_status, 
        POS_CASH_balance_status, 
        installments_payments_status,
    ]
    
    for status in status_list:
        if status == "File uploaded correctly":
            upload_counter += 1
    
    # If files uploaded correctly, we make predictions and diplays results
    if upload_counter == 7:
    
        # Preprocessing
    
        # Cleaning, feature engineering and merge
        uploaded_datas, _ = preprocess_joint(
            application_df, 
            bureau_balance_df, 
            bureau_df, 
            previous_application_df, 
            credit_card_balance_df, 
            POS_CASH_balance_df, 
            installments_payments_df, 
            imputer=imputer,
        )

        # Preprocessing for model (standardization, encoding, imputation)
        x_upload, y_upload, _ = final_preprocessing(
            uploaded_datas, 
            is_balance=False, 
            existing_preprocessor=preprocessor, 
            is_existing_cols=True, 
            full_cols=features_list
        )

        x_upload.to_csv('./uploaded_datas/data_1.csv')
        selected_customer = x_upload.iloc[0]
        
        # Making predictions with API
        PREDICT_URI = 'https://p7-openclassrooms-api.herokuapp.com/api/predict-new/'

        with open('./uploaded_datas/data_1.csv', "rb") as file_1:
            file_dict = {'data' : file_1}
            response = requests.request(method='POST', url=PREDICT_URI, files=file_dict)

        dict_response = json.loads(response.content.decode('utf-8'))

        probas = dict_response['datas']['proba']
        predictions = dict_response['datas']['prediction']

        outputs_pred = np.array(['Accepted','Refused'])
        
        current_prediction = outputs_pred[predictions]

        fig = generate_shap_local_feat(explainer, selected_customer, features_list, layout_dict)
    
    # If files not uploaded correctly, we display empty graphs and waiting text
    else:
        
        current_prediction = "Waiting for uploaded files"
        probas = 0
        fig = px.bar(x=[0], y=[0])
        fig.update_layout(layout_dict)     
        
    return application_status, bureau_balance_status, bureau_status, previous_application_status, credit_card_balance_status, POS_CASH_balance_status, installments_payments_status, current_prediction, probas, fig

# ----------------------------------------------------------------------------------
# Functions and variables tab 2 (predict new)
            
# Style for upload buttons
style_dict={
    'width': '160px',
    'height': '60px',
    # 'lineHeight': '60px',
    # 'borderWidth': '1px',
    # 'borderStyle': 'dashed',
    # 'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px',
}
            
def decode_csv_to_df(content):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    return df
    
def test_upload(content, expected_shape):
        
    if content is not None:    
        
        df = decode_csv_to_df(content)
        
        if df.shape[1] != expected_shape:
            status = "Your file should contain {} columns (y.c. index), please reload correct file".format(expected_shape)
        else:
            status = "File uploaded correctly"
    
    else:
        status = "File not uploaded"
        df = None
        
    return df, status
    
# ----------------------------------------------------------------------------------
# Layout tab 3 (model description)

def build_model_description():

    return (
        html.Div(id="status-tab-3-container", children=[
            
            # Left panel with 1 graph
            html.Div(id='quick-stats', children=[
                html.Div(id="banner-graph-1", className="section-banner", children=[dcc.Markdown("Global feature importance")]),
                dcc.Graph(
                    id="global-feat-importance",
                    figure=generate_shap_global_feat(shap_global_top_features, layout_dict_global)
                ),
            ]),

            # Right panel with 3 graphs
            html.Div(id="tab-3-graphs-container",children=[

                html.Div(id="tab-3-graph-1", children=[
                    html.Div(id="banner-graph-1", className="section-banner", children=[
                        dcc.Markdown(id="banner-graph-1-text"),
                        dcc.Dropdown(top_global_features, value=top_global_features[0], id='feat-1-dropdown'),
                    ]),
                    dcc.Graph(id="distrib-feat-1"),
                ]),

                html.Div(id="tab-3-graph-2", children=[
                    html.Div(id="banner-graph-2", className="section-banner", children=[
                        dcc.Markdown(id="banner-graph-2-text"),
                        dcc.Dropdown(top_global_features, value=top_global_features[1], id='feat-2-dropdown'),
                    ]),
                    dcc.Graph(id="distrib-feat-2"),
                ]),

                html.Div(id="tab-3-graph-3", children=[
                    html.Div(id="banner-graph-3", className="section-banner"),
                    dcc.Graph(id="scatter-feat-1-2"),
                ]),
            ]),
        ]),
    )
    
# ----------------------------------------------------------------------------------
# Callbacks tab 3 (model description)
    
@app.callback(
    Output('banner-graph-1-text', 'children'),
    Output('banner-graph-2-text', 'children'),
    Output('banner-graph-3', 'children'),
    Output('distrib-feat-1', 'figure'),
    Output('distrib-feat-2', 'figure'),
    Output('scatter-feat-1-2', 'figure'),
    Input('feat-1-dropdown', 'value'),
    Input('feat-2-dropdown', 'value'),
)

def plot_graphs(feat_1, feat_2):
    
    # Name container banner 
    title_graph_1 = "Distribution of {}".format(feat_1)
    title_graph_2 = "Distribution of {}".format(feat_2)
    title_graph_3 = "Scatter of {} and {}".format(feat_1, feat_2)
    
    # Request database for variables values
    feature_1_dist = find_feature_distribution(feat_1)
    feature_2_dist = find_feature_distribution(feat_2)
    
    # Plot graphs
    colors = ['#f4d44d', '#EC8484']
    
    feat_1_fig = generate_hist_graph(feat_1, feature_1_dist, y, colors)
    feat_2_fig = generate_hist_graph(feat_2, feature_2_dist, y, colors)
    scatter_fig = generate_scatter_graph(feat_1, feature_1_dist, feat_2, feature_2_dist, y, colors)
    
    return title_graph_1, title_graph_2, title_graph_3, feat_1_fig, feat_2_fig, scatter_fig
    
# ----------------------------------------------------------------------------------
# Functions and variables tab 3 (model description)

# Modify layout for global feat
layout_dict_global = layout_dict.copy()
layout_dict_global['autosize'] = False
layout_dict_global['height'] = 700
layout_dict_global['margin'] = {'l':110, 'b':90, 't':10}
# layout_dict_global['yaxis'] = {'title': {'text' : None}}
# layout_dict_global['xaxis'] = {'title': {'text' : None}}

# Modify layout for histograms
layout_dict_hists = layout_dict.copy()
layout_dict_hists['autosize'] = False
layout_dict_hists['height'] = 180
layout_dict_hists['margin'] = {'l':25, 'r':10, 'b':0, 't':0}
layout_dict_hists['yaxis'] = {'visible': False}
layout_dict_hists['xaxis'] = {'visible': False}

# Modify layout for scatter
layout_dict_scatter = layout_dict.copy()
layout_dict_scatter['autosize'] = False
layout_dict_scatter['height'] = 180
layout_dict_scatter['margin'] = {'l':25, 'r':10, 'b':0, 't':0}
layout_dict_scatter['legend'] = {'title': None}
layout_dict_scatter['yaxis'] = {'visible': False}
layout_dict_scatter['xaxis'] = {'visible': False}

# Variables for dropdowns
top_global_features = shap_global_top_features["Feature name"]

# Shap global feature importance
def generate_shap_global_feat(shap_global_top_features, layout_dict):
    
#     # Shap values for features
#     shap_df_global = pd.DataFrame(abs(shap_values_global), columns=x.columns)
#     top_global_features = shap_df_global.mean(axis=0).sort_values(ascending=False)[:20].index
#     top_features_values = shap_df_global.mean(axis=0).sort_values(ascending=False)[:20]
    
#     # Generate figure to plot
#     plot_df = pd.DataFrame()
#     plot_df["Feature name"] = top_global_features
#     plot_df["Shap values"] = top_features_values.values
#     plot_df = plot_df.sort_values(by="Shap values")
    
    fig = px.bar(shap_global_top_features, x="Shap values", y="Feature name")
    fig.update_layout(layout_dict)
    fig.update_traces(
        marker_color='#f4d44d', 
        marker_line_color='darkgrey',
        marker_line_width=1, opacity=1
    )
    
    return fig

def generate_hist_graph(feature, feature_dist, y, colors):
    
    # Create df with feature
    df = pd.DataFrame()
    df[feature] = feature_dist
    df['TARGET'] = y['TARGET'].values
    
    x1 = df[df['TARGET'] == 0][feature].values
    x2 = df[df['TARGET'] == 1][feature].values

    hist_data = [x1, x2]

    group_labels = ['Accepted', 'Refused']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels, colors=colors, bin_size=.2, show_rug=False)
    fig.update_layout(layout_dict_hists)
    
    return fig

def generate_scatter_graph(feature_1, feature_1_dist, feature_2, feature_2_dist, y, colors):
    
    y_bis = y.copy()
    y_bis['TARGET_str'] = y_bis['TARGET'].apply(lambda x: "Accepted" if x == 0 else "Refused")
    
    # Create df with feature
    df = pd.DataFrame()
    df[feature_1] = feature_1_dist
    df[feature_2] = feature_2_dist
    df['TARGET'] = y_bis['TARGET_str'].values
    
    fig = px.scatter(df, x=feature_1, y=feature_2, color="TARGET", color_discrete_sequence=colors, labels=None)
    fig.update_layout(layout_dict_scatter)
    
    return fig

    
if __name__ == '__main__':
    app.run_server(debug=True, port=5000)