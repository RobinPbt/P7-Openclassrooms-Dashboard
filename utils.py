from models import engine
import pandas as pd

def find_customer_features(id_customer):
    customer_features = engine.execute("""SELECT * FROM x_train WHERE "SK_ID_CURR" = {}""".format(id_customer)).fetchall()
    customer_features = pd.Series(customer_features[0])
    customer_features.drop('SK_ID_CURR', inplace=True)
    
    return customer_features

def find_feature_distribution(feature_name):
    feature_distribution = engine.execute("""SELECT "{}" FROM x_train""".format(feature_name)).fetchall()
    feature_distribution = [i[0] for i in feature_distribution]
    
    return feature_distribution

def get_features_names():
    feature_names = engine.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='x_train'").fetchall()
    feature_names = [i[0] for i in feature_names]
    feature_names.remove('SK_ID_CURR')
    
    return feature_names

def get_full_train_set():
    full_train_set = engine.execute("SELECT * FROM x_train").fetchall()
    full_train_set = pd.DataFrame(full_train_set)
    full_train_set.index = full_train_set['SK_ID_CURR']
    full_train_set.drop('SK_ID_CURR', axis=1, inplace=True)
    
    return full_train_set

def get_full_labels():
    full_labels = engine.execute("SELECT * FROM y_train").fetchall()
    full_labels = pd.DataFrame(full_labels)
    full_labels.index = full_labels['SK_ID_CURR']
    full_labels.drop('SK_ID_CURR', axis=1, inplace=True)
    
    return full_labels