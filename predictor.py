#Import 
import pandas as pd
import numpy as np
import trees
import svm

#Import Decision tree functions
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#Import SVM and data processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import _data
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.impute import SimpleImputer

#Create dataframe
data = 'data/NCAAHistoricalData.csv'
df = pd.read_csv(data)
validate_df = df[3906:]
df = df[:3906]

#Trim dataframe/change string values to int
def format(data):
    data.pop('index')
    data.pop('Year')
    data.pop('Team1')
    data.pop('spread')
    data.pop('Team2')
    data.pop("Score1")
    data.pop("Score2")

    winner_encoder = LabelEncoder()
    winner_encoder.fit(df["winner"])
    df["winner"] = winner_encoder.transform(df["winner"])

    return data

#Split data to train and test sets
def split(data):
    train_x, test_x, train_y, test_y= None,None,None,None
    train_x, test_x, train_y, test_y = train_test_split( data[:3906, :49], data[:3906, 49], test_size = .2, train_size = .8, shuffle = False)
    return (train_x, train_y, test_x, test_y)

#Normalize the data
def normalize(data):
    scaler = MinMaxScaler()

    scaler.fit(data)
    data = scaler.transform(data)

    return data

#PCA
def pca(input_data, dimensions):
    dimensions_extra = dimensions + 1
    pca = PCA(n_components = dimensions_extra)
    reduced_data = pca.fit_transform(input_data)
    return reduced_data

#ICA
def ica(input_data, dimensions):
    dimensions_extra =  dimensions + 1
    ica = FastICA(n_components = dimensions_extra)
    reduced_data = ica.fit_transform(input_data)
    return reduced_data


if __name__ == "__main__":
    #Manipulating data for testing
    format(df)
    imp = SimpleImputer(missing_values= np.nan, strategy='mean')
    imp.fit(df[1140:])
    df = imp.transform(df)
    train_data, train_labels, test_data, test_labels = split(df)
    train_data = normalize(train_data)
    test_data = normalize(test_data)
    
    #Random forest
    print("Metrics for Random Forest")
    trees.get_rf_metrics(train_data, train_labels, test_data, test_labels)
    
    #SVM
    print("Metrics for SVM")
    svm.get_svm_metrics(train_data, train_labels, test_data, test_labels)

    #PCA both
    pca_dimensions = 20
    pca_train_x, pca_train_y, pca_test_x, pca_test_y = split(df)
    pca_train_x = pca(pca_train_x, pca_dimensions)
    pca_test_x = pca(pca_test_x, pca_dimensions)
    normalize(pca_train_x)
    normalize(pca_test_x)

    print("Metrics for PCA reduction of Random frest and SVM, respectively")
    trees.get_rf_metrics(pca_train_x, pca_train_y, pca_test_x, pca_test_y)
    svm.get_svm_metrics(pca_train_x, pca_train_y, pca_test_x, pca_test_y)

    #ICA both
    ica_dimensions = 20
    ica_train_x, ica_train_y, ica_test_x, ica_test_y = split(df)
    ica_train_x = ica(ica_train_x, ica_dimensions)
    ica_test_x = ica(ica_test_x, ica_dimensions)
    normalize(ica_train_x)
    normalize(ica_test_x)

    print("Metrics for ICA altered Random forest and SVM, respectively")
    trees.get_rf_metrics(ica_train_x, ica_train_y, ica_test_x, ica_test_y)
    svm.get_svm_metrics(ica_train_x, ica_train_y, ica_test_x, ica_test_y)
