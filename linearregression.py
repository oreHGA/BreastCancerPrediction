import pandas as pd
import numpy
import sklearn as scikit

def main():
    location = "breastcancerdataset.csv"
    dataset = pd.read_csv(location, ',')
    labels = dataset.iloc[:, 1].values 
    features = dataset.iloc[:, 2:].values

    # now we need to perform feature scaling on the features 

    print labels


main()