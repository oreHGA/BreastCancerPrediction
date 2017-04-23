import pandas as pd
import numpy
import sklearn as scikit
import sklearn.preprocessing

def main():
    location = "breastcancerdataset.csv"
    dataset = pd.read_csv(location, ',')
    labels = dataset.iloc[:, 1].values
    features = dataset.iloc[:, 2:].values

    # now we need to perform feature scaling on the features 
    features = scikit.preprocessing.normalize(features, 'l1')
    
    # so from the 570 samples , 470 will be used for training and then 100 for testing
    train_x = features[1:470, ]
    train_y = labels[1:470]
    test_x = features[471:, ]
    test_y = labels[471:]
    # now that we have our normalized training and testing data .. we need to go ahead and start the calculations
    print train_x


main()