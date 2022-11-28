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
df = 