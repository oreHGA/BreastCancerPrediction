import pandas as pd
import numpy
import sklearn as scikit
import sklearn.preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def main():
    location = "breastcancerdataset.csv"
    dataset = pd.read_csv(location, ',')
    labels = dataset.iloc[:, 1].values
    temp = labels
    count = 0
    for i in labels:
        if (i == 'M'):
            temp[count] = '1'
        else:
            temp[count] = '0'
        count = count + 1
    features = dataset.iloc[:, 2:].values

    # now we need to perform feature scaling on the features
    features = scikit.preprocessing.normalize(features, 'l1')
    train_x = features[0:470, ]
    train_y = temp[0:470]
    test_x = features[470:, ]
    test_y = temp[470:]

    # now we implement logistic regression
    # train_y = numpy.ravel(train_y)

    model_A = LogisticRegression(C=1e6,solver='liblinear')
    model_A = model_A.fit(train_x, train_y)

    prediction = model_A.predict(test_x)

    accuracy = metrics.accuracy_score(test_y,prediction)

    # now we generate the confusion matrix
    print "Your logistic regression algorithm has an accuracy of ->"
    print accuracy

main()
