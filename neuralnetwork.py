import pandas as pd
import numpy
import sklearn as scikit
import sklearn.preprocessing
from sklearn import metrics

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    location = "breastcancerdataset.csv"
    dataset = pd.read_csv(location, ',')
    labels = dataset.iloc[:, 1].values
    temp = labels
    count = 0
    for i in labels:
        if (i=='M'):
            temp[count] = '1'
        else:
            temp[count] = '0'
        count = count + 1
    features = dataset.iloc[:, 2:].values

    # now we need to perform feature scaling on the features
    features = scikit.preprocessing.normalize(features,'l1')
    train_x = features[0:470, ]
    train_y = temp[0:470]
    test_x = features[470: , ]
    test_y = temp[470: ]

    # Making the neural network model
    # sklearn.neural_network.
    model_A = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(376,282,188,94))
    model_A.fit(train_x,train_y)

    predictions = model_A.predict(test_x)

    print(confusion_matrix(test_y, predictions))
    print(classification_report(test_y, predictions))

    accuracy = metrics.accuracy_score(test_y, predictions)
    print accuracy

main()
