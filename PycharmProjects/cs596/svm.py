import pandas as pd
import numpy as np
import sklearn as scikit
import sklearn.preprocessing
from sklearn import svm
from sklearn import metrics
import sklearn.utils.multiclass as checking
import matplotlib.pyplot as plt


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
    #features = scikit.preprocessing.normalize(features, 'l1')  #normalizing the features decreased accuracy by over 20%
    train_x = features[0:470, ]
    train_y = temp[0:470]
    test_x = features[470:, ]
    test_y = temp[470:]

    feature1 = features[:,0]    # radius_mean column
    feature2 = features[:,1]    # texture_mean column
    X = np.vstack((feature1, feature2)).T

    clf = svm.SVC(kernel='linear')
    clf.fit(X,temp)
    prediction = clf.predict(X)

    index = 0
    for i in X:
        if(temp[index] == "1"):
            plt.scatter(X[index,0],X[index,1], c='red', marker='+')
        else:
            plt.scatter(X[index,0],X[index,1], c='green', marker='.')
        index += 1

    accuracy = metrics.accuracy_score(temp, prediction)
    print "accuracy: ",accuracy

    w = clf.coef_[0]

    a = -w[0] / w[1]

    xx = np.linspace(0, 20)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    h0 = plt.plot(xx, yy, 'k-')

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.show()

main()