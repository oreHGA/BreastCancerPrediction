import pandas as pd
import numpy
import sklearn as scikit
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def plotfeatures(xsample,ysample):
    count = 0
    for i in ysample:
        if i == '1':
            m, = plt.plot(xsample[count,1], xsample[count,2],'ro')
        elif i == '0': 
            b, = plt.plot(xsample[count,1], xsample[count,2],'bo')
        plt.hold(True)
        count += 1 
    plt.xlabel('texture_mean')
    plt.ylabel('radius_mean')
    plt.legend(handles=[m ,b],labels = ['Malignant','Benign'])
    plt.title('A scatter plot of the Dataset considering 2 features, radius_mean and texture_mean')
    plt.show()

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
    #  looks like we might have to further split the dataset into training and validation data
    # now we implement logistic regression
    plotfeatures(xsample=features,ysample=labels)
    model_A = LogisticRegression(C=1e6,solver='liblinear')
    model_A = model_A.fit(train_x, train_y)
    prediction = model_A.predict(test_x)

    accuracy_A = metrics.accuracy_score(test_y,prediction)
    conf_matrix_A = metrics.confusion_matrix(test_y,prediction)
    eval_A = metrics.classification_report(test_y,prediction)
    # now we generate the confusion matrix
    print "Your logistic regression algorithm has an accuracy of ->"
    print accuracy_A
    print conf_matrix_A
    print eval_A

    #  next step is to start plotting the graphs showing the error rates

main()

