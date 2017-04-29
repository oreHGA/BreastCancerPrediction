import pandas as pd
import numpy
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def plotfeatures(xsample, ysample):
    count = 0
    for i in ysample:
        if i == '1':
            malignant, = plt.plot(xsample[count, 1], xsample[count, 2], 'ro')
        elif i == '0':
            benign, = plt.plot(xsample[count, 1], xsample[count, 2], 'bo')
        plt.hold(True)
        count += 1
    plt.xlabel('texture_mean')
    plt.ylabel('radius_mean')
    plt.legend(handles=[malignant, benign], labels=['Malignant', 'Benign'])
    plt.title('A scatter plot of the Dataset considering 2 features, radius_mean and texture_mean')
    plt.show()

def getdataset(location="./data/breastcancerdataset.csv"):
    dataset = pd.read_csv(location, ',')
    labels = dataset.iloc[:, 1].values
    count = 0
    for i in labels:
        if i == 'M':
            i = '1'
        else:
            i = '0'
        count = count + 1
    features = dataset.iloc[:, 2:].values
    features = sklearn.preprocessing.normalize(features, 'l1')
    train_x = features[0:235, ]
    train_y = labels[0:235]
    valid_x = features[235:470, ]
    valid_y = labels[235:470]
    test_x = features[470:, ]   # we do not touch these variables until the we get the best model
    test_y = labels[470:]       # we do not touch these variables until the we get the best model
    #plotfeatures(xsample=features, ysample=labels)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def logistic():
    train_x, train_y, valid_x, valid_y, test_x, test_y = getdataset()
    # First Possible Model
    model_a = LogisticRegression(C=1e6, solver='liblinear')
    model_a = model_a.fit(train_x, train_y)
    prediction = model_a.predict(valid_x)
    accuracy_a = metrics.accuracy_score(valid_y, prediction)
    # next possible model
    model_b = LogisticRegression(C=1e6, solver='sag')
    model_b = model_b.fit(train_x, train_y)
    prediction = model_b.predict(valid_x)
    accuracy_b = metrics.accuracy_score(valid_y, prediction)

    if accuracy_a > accuracy_b:
        print "A is more accurate"
        print accuracy_a
    else:
        print "B is more accurate"
        print accuracy_b


    # we'll do this after getting the best model
    #conf_matrix_A = metrics.confusion_matrix(test_y, prediction)
    #eval_A = metrics.classification_report(test_y, prediction)
    # now we generate the confusion matrix
    #  next step is to start plotting the graphs showing the error rates

logistic()

