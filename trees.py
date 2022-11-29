from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def get_rf_metrics(train_data,train_labels,test_data,test_labels):
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(train_data,train_labels)

    pred = rf.predict(test_data)

    accuracy = accuracy_score(test_labels,pred)
    precision = precision_score(test_labels,pred, average= 'micro')
    recall = recall_score(test_labels,pred, average='micro')

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
