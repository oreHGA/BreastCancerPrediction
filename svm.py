import pandas as pd
import numpy
import sklearn as scikit
import sklearn.preprocessing

def main():
    location = "breastcancerdataset.csv"
    dataset = pd.read_csv(location, ',')
    labels = dataset.iloc[:, 1].values 
    temp = labels
    count = 0
    for i in labels:
        if (i=='M'):
            temp[count] = 1
        else:
            temp[count] = 0
        count = count + 1
    features = dataset.iloc[:, 2:].values

    # now we need to perform feature scaling on the features 
    features = scikit.preprocessing.normalize(features,'l1')
    train_x = features[0:470, ]
    train_y = temp[0:470]
    test_x = features[470: , ]
    test_y = temp[470: ]
    print test_y


main()
