#Import 
import pandas as pd
import numpy as np
import math

#Import Decision tree functions
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#Import SVM and data processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import _data
from sklearn.svm import SVC

#Create dataframe
data = 'data/NCAAHistoricalData.csv'
df = pd.read_csv(data)

#Trim dataframe/change string values to int
def format(data):
    data.pop('index')
    data.pop('Year')
    data.pop('Team1')
    data.pop('Team2')
    data.pop("winner")
    data.pop("Score1")
    data.pop("Score2")
    data.pop("G1")
    data.pop("G2")

    return data

#Split data to train and test sets
def split(data):
    train_x, test_x, train_y, test_y= None,None,None,None
    train_x, test_x, train_y, test_y = train_test_split( data.values[:4032, :47], data.values[:4032, 47], test_size = .2, train_size = .8, shuffle = False)
    return (train_x, train_y, test_x, test_y)

##Normalize the data
def normalize(data):
    scaler = MinMaxScaler()

    scaler.fit(data)

    data = scaler.transform(data)

    return data, scaler

#PCA/ICA

#Retry both methods of training

if __name__ == "__main__":
    print(format(df))
    train_x, train_y, test_x, test_y = split(df)
    print(train_x)
    print(test_y)
