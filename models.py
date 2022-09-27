from flask import Flask
from sklearn.model_selection import train_test_split
from config import *

import os
import pandas as pd
import flask_sqlalchemy
import sqlite3

# SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
# Need to replace postgres by postgresql
# SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.split("//")
# SQLALCHEMY_DATABASE_URI[0] = "postgresql:"
# SQLALCHEMY_DATABASE_URI = "".join(SQLALCHEMY_DATABASE_URI)

SQLALCHEMY_DATABASE_URI = "postgresql://hvoslnoxsyjbke:3af51c18a7e655cba16387d6da046249a43b4a99224b77892d3955e0b35c9b66@ec2-63-32-248-14.eu-west-1.compute.amazonaws.com:5432/d4j0hehuo81ve3"

# Create engine object
engine = flask_sqlalchemy.sqlalchemy.engine.create_engine(SQLALCHEMY_DATABASE_URI)

#----------------------------------------------------------------------------------------
# Following instructions performed only once to create database

# # Load datas we will use for our database
# x = pd.read_csv('../Projet 7/Clean_datas/real_x.csv')
# x.index = x['SK_ID_CURR']

# y = pd.read_csv('../Projet 7/Clean_datas/real_y.csv')
# y.index = x['SK_ID_CURR']

# x.drop('SK_ID_CURR', axis=1, inplace=True)
# y.drop('Unnamed: 0', axis=1, inplace=True)

# # # If we want to reduce RAM and memory consumption
# x, _, y, _ = train_test_split(x, y, train_size=SUBSET_SIZE, random_state=RANDOM_STATE)

# # # Link database file with these datas
# x.to_sql('x_train', con=engine, if_exists='fail', index=True, index_label='SK_ID_CURR')
# y.to_sql('y_train', con=engine, if_exists='fail', index=True, index_label='SK_ID_CURR')