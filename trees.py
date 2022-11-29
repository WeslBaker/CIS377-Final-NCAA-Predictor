from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from webbrowser import get
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from predictor import df

#PATH = 'NCAAHistoricalData.csv'

def get_data():
    #df = pd.read_csv(PATH)
    X = df.values[:,1:]
    Y = df.values[:,0]
    
    train_x, test_x, train_y, test_y = train_test_split(X,Y,train_size=0.8,test_size=0.2,shuffle=True)
    return train_x, test_x, train_y, test_y
    

def normalize(data):
    scaler = MinMaxScaler()

    scaler.fit(data)

    data = scaler.transform(data)

    return data, scaler

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = get_data()

    normalize(train_x)

def get_metrics(model,train_data,train_labels,test_data,test_labels,algorithm_title):
    model.fit(train_data,train_labels)

    # norm test data
    test_data = scaler.transform(test_data)
    pred = model.predict(test_data)

    accuracy = accuracy_score(test_labels,pred)
    precision = precision_score(test_labels,pred, average= 'micro')
    recall = recall_score(test_labels,pred, average='micro')

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = get_data()

    norm_data, scaler = normalize(train_x)

    rf = RandomForestClassifier(n_estimators=50)

    print('Random Forest:')
    get_metrics(rf,norm_data,train_y,test_x,test_y,'rf')
