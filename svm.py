#Import general
import pandas as pd
import numpy as np
import math

#Import SVM and data processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import _data
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#def normalize(historical_data):
#    df = historical_data
    

from sklearn.model_selection import train_test_split
from predictor import df

from sklearn.metrics import accuracy_score, precision_score, recall_score

train_x, test_x, train_y, test_y= None,None,None,None

df['Exited'] = LabelEncoder().fit_transform(df['Exited'])
X = df.values[:,1:]
Y = df.values[:,0]
train_x, test_x, train_y, test_y = train_test_split(X, Y,train_size=0.8,test_size=0.2,random_state=42,shuffle=True)
#Normalize data
#we could standardize the spliting and normalization not sure which is better
scaler = None
original_train = train_x
############################STUDENT CODE GOES HERE#########################
# assign the scaler to an instance of StandarScaler
scaler = StandardScaler()
# Fit the scaler and tranform the data. The train_x variable should now be normalized
scaler.fit(original_train)
original_train = scaler.transform(original_train)
#################################END CODE##################################
assert(type(scaler) == _data.StandardScaler)
assert(np.array_equal(original_train,train_x) == False)
#Train SVM
support_vector_classifier = SVC(kernel='rbf')
support_vector_classifier.fit(train_x, train_y)
#Test SVM
original_test = test_x
original_test = scaler.transform(original_test)
assert(np.array_equal(original_test,test_x) == False)


### this code failed for me so it may need work, could be from not having the right libaries insalled
results_dict = {'Accuracy':0,'Precision':0,'Recall':0}
############################STUDENT CODE GOES HERE#########################
# use the SVM to predict on the test data
pred = support_vector_classifier.predict(original_test)
# compare the truth labels with the predicted labels for accuracy, precision, and recall
# store the results into the dataframe
print(accuracy_score(test_y,pred))
print(precision_score(test_y,pred, average = 'micro'))
print(recall_score(test_y,pred, average='micro'))
results_dict['Accuracy'] = accuracy_score(test_y,pred)
results_dict['Precision'] = precision_score(test_y,pred, average='micro')
results_dict['Recall'] = recall_score(test_y,pred, average='micro')
#################################END CODE##################################

assert(results_dict['Accuracy'] > 0.15)
assert(results_dict['Precision'] > 0.05)
assert(results_dict['Recall'] > 0.30)
results_dict
