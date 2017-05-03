import pandas as pd
import numpy
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neural_network import MLPClassifier

def plotfeatures(xsample, ysample):
    count = 0
    for i in ysample:
        if i == 'M':
            malignant, = plt.plot(xsample[count, 1], xsample[count, 2], 'ro')
        elif i == 'B':
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
    temp = []
    for i in labels:
        if i == 'M':
            temp.append('0')
        else:
            temp.append('1')
        count = count + 1
    features = dataset.iloc[:, 2:].values
    return features,labels,temp     #temp is the discrete value of the labels

def preprocess_data(X,Y):
    feat = sklearn.preprocessing.normalize(X, 'l1')
    train_x = feat[0:235, ]
    train_y = [map(int, x) for x in Y[0:235]]
    valid_x = feat[235:470, ]
    valid_y = [map(int, x) for x in Y[235:470]]
    test_x = feat[470:, ]   # we do not touch these variables until the we get the best model
    test_y = [map(int, x) for x in Y[470:]]   # we do not touch these variables until the we get the best model
    return train_x, train_y, valid_x, valid_y, test_x, test_y,feat


def logistic(train_x, train_y, valid_x, valid_y):
    # First Possible Model
    model_a = LogisticRegression(C=1e6, solver='liblinear')
    train_scores, valid_scores = validation_curve(model_a, train_x, train_y, "alpha",numpy.logspace(-7, 3, 3))
    train_sizes, train_scores, test_scores  = learning_curve(model_a, train_x,train_y, cv=None, n_jobs=1, train_sizes=numpy.linspace(.1, 1.0, 5))
   
    # model_a = model_a.fit(train_x, train_y)
    prediction = model_a.predict(valid_x)
    accuracy_a = metrics.accuracy_score(valid_y, prediction)
    #next possible model
    model_b = LogisticRegression(C=1e6, solver='sag')
    model_b = model_b.fit(train_x, train_y)
    prediction = model_b.predict(valid_x)
    accuracy_b = metrics.accuracy_score(valid_y, prediction)
    if accuracy_a >= accuracy_b:
        better_model = model_a
    else:
        better_model = model_b
    return model_a

def neuralnetwork(train_x, train_y, valid_x, valid_y):
    #First Possibile Model
    model_a = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(376,282,188,94))
    model_a.fit(train_x,train_y)
    prediction = model_a.predict(valid_x)
    accuracy_a = metrics.accuracy_score(valid_y, prediction)
    # next possible model
    model_b = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(300,200,100,10))
    model_b.fit(train_x,train_y)
    prediction = model_b.predict(valid_x)
    accuracy_b = metrics.accuracy_score(valid_y, prediction)
    if accuracy_a >= accuracy_b:
        better_model = model_a
    else:
        better_model = model_b
    return better_model

def supportvectors():
    pass

def kmeansimplementation(train_x, train_y, valid_x, valid_y):
    model_a = KMeans(n_clusters=2, init='k-means++', tol=1e-6, random_state=0).fit(train_x)
    prediction = model_a.predict(valid_x)
    accuracy_a = metrics.accuracy_score(valid_y, prediction)
    # next possible model
    # learning_curve.
    model_b = KMeans(n_clusters=2, init='random', tol=1e-6, random_state=0).fit(train_x)
    prediction = model_b.predict(valid_x)
    accuracy_b = metrics.accuracy_score(valid_y, prediction)
    if accuracy_a >= accuracy_b:
        better_model = model_a
    else:
        better_model = model_b
    
    #  to plot the learned cluster
    fig = plt.figure(1, figsize=(4,3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    labels = better_model.labels_
    ax.scatter(train_x[:, 1], train_x[:, 2], train_x[:, 3], c=labels.astype(numpy.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Radius Mean')
    ax.set_ylabel('Texture Mean')
    ax.set_zlabel('Perimeter Mean')
    plt.show()

    return better_model

#  we would then see how they do with the testing data
features,labels,temp = getdataset()

train_x, train_y, valid_x, valid_y, test_x, test_y, features = preprocess_data(features, temp)

# to visualize the dataset to the users
# plotfeatures(xsample=features, ysample=labels)

# getting the best models for testing on the testing dataset
# logistic_model = logistic(train_x,train_y,valid_x,valid_y)
kmeans_model = kmeansimplementation(train_x,train_y,valid_x,valid_y)
nn_model = neuralnetwork(train_x,train_y,valid_x,valid_y)