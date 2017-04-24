import pandas as pd
import numpy as np
import sklearn as scikit
import sklearn.preprocessing
from sklearn import svm
from sklearn import metrics 
import sklearn.utils.multiclass as checking

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
    #print(checking.check_classification_targets(test_y))
    model = svm.SVC()
    model.fit(train_x,train_y)
    prediction=model.predict(test_x)
    y = metrics.accuracy_score(test_y,prediction)
    print(y)


main()
