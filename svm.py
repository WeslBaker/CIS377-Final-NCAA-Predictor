#Import general
import pandas as pd
import numpy as np
import math

#Import SVM and data processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import _data
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder   

from sklearn.model_selection import train_test_split
from predictor import df

from sklearn.metrics import accuracy_score, precision_score, recall_score

def get_svm_metrics(train_x, train_y, test_x, test_y):
    svm = SVC(kernel='linear')
    svm.fit(train_x, train_y)

    # use the SVM to predict on the test data
    pred = svm.predict(test_x)

    # compare the truth labels with the predicted labels for accuracy, precision, and recall
    # store the results
    accuracy = accuracy_score(test_y,pred)
    precision = precision_score(test_y,pred, average='micro')
    recall = recall_score(test_y,pred, average='micro')
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
