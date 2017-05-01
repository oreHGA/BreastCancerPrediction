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
    X = np.vstack((feature1, feature2)).T   #create single array with 2 columns each; used for fitting the model using clf.fit()

    #plot classifications
    index = 0
    for i in X:
        if(temp[index] == "1"):
            plt.scatter(X[index,0],X[index,1], c='red', marker='+')
        else:
            plt.scatter(X[index,0],X[index,1], c='green', marker='.')
        index += 1



    # #create a linear plot
    # clf = svm.SVC(kernel='linear')
    # clf.fit(X,temp)
    # prediction = clf.predict(X)
    #
    # #get separating hyperplane
    # w = clf.coef_[0]
    # a = -w[0] / w[1]
    # xx = np.linspace(0, 20)
    # yy = a * xx - (clf.intercept_[0]) / w[1]
    # h0 = plt.plot(xx, yy, 'k-')
    #
    # #plot the parallels to the separating hyperplane
    # margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    # yy_down = yy + a * margin
    # yy_up = yy - a * margin
    # plt.plot(xx, yy, 'k-')
    # plt.plot(xx, yy_down, 'k--')
    # plt.plot(xx, yy_up, 'k--')

    #Create multilinear plot
    clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
    clf.fit(X, temp)
    prediction = clf.predict(X)

    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    accuracy = metrics.accuracy_score(temp, prediction)
    print "accuracy: ", accuracy

    plt.show()



main()