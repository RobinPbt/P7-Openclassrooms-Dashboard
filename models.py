from flask import Flask
from sklearn.model_selection import train_test_split
from config import *

import os
import pandas as pd
import flask_sqlalchemy

SQLALCHEMY_DATABASE_URI = "postgresql://hjfjmqghwjxjcg:62c01f3030a04e47d89be74a5105c4a6ddfafba045d1694279a3098466f8b707@ec2-52-49-201-212.eu-west-1.compute.amazonaws.com:5432/d6cku0lf8ftp7"

# Create engine object
engine = flask_sqlalchemy.sqlalchemy.engine.create_engine(SQLALCHEMY_DATABASE_URI)

#----------------------------------------------------------------------------------------
# Following instructions performed only once to create and fill database

# # Load datas we will use for our database
# x = pd.read_csv('../Projet 7/Clean_datas/real_x.csv')
# x.index = x['SK_ID_CURR']

# y = pd.read_csv('../Projet 7/Clean_datas/real_y.csv')
# y.index = x['SK_ID_CURR']

# x.drop('SK_ID_CURR', axis=1, inplace=True)
# y.drop('Unnamed: 0', axis=1, inplace=True)

# # If we want to reduce RAM and memory consumption
# x, _, y, _ = train_test_split(x, y, train_size=SUBSET_SIZE, random_state=RANDOM_STATE)

# # Link database file with these datas
# x.to_sql('x_train', con=engine, if_exists='fail', index=True, index_label='SK_ID_CURR')
# y.to_sql('y_train', con=engine, if_exists='fail', index=True, index_label='SK_ID_CURR')