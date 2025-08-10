import pandas as pd
from logger import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder


def get_data():
    data=pd.read_csv('artifacts/iris.csv')
    # data Preprocessing
    label_encoder = LabelEncoder()
    #preprocessing
    data['species'] = label_encoder.fit_transform(data['species'])
    X=data.drop(columns=['species'])
    y=data["species"]
    
    print(data.head())
    logging.info("Iris Data loaded successfully.")
    return X,y



