import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True) # setting the printing options 

# Load data from the data file by using pandas and convert it to numpy array
def importData(filename: str):
    # Import data from the .data file by using pandas 
    # # read_fwf() reads a table of fixed-width formatted lines into DataFrame
    # data = pd.read_fwf(filename, header=0)
    data = pd.read_csv(filename, header=0)
    # Convert the dataset to a numpy array
    data_npy = np.asarray(data)
    return data_npy


# Change the class name strings to numbers. Then split the data to dataset and clas labels
# For more details, please check the report for class distribution.
def manageData(data: np.ndarray, data_class: list):
    # Change all 10 data classes from string to number
    for i in range(len(data_class)):
        data[data == data_class[i]] = i
    # # Sort the data by the last column of the data matrix
    # sortedData = data[data[:, -1].argsort()]
    dataset = np.array(data[:, :-1], dtype=float)
    labels = np.array(data[:, -1:], dtype=int)
    return dataset, labels


# Shuffle the data by using the magical number 42, 
# then separate the data by 70% for training, 30% for testing
def trainTestSplit(data, labels):
    training_data, testing_data, training_labels, testing_labels \
        = train_test_split(data,labels,test_size=0.3,train_size=0.7,random_state=42)
    return training_data, testing_data, training_labels, testing_labels


# Load data, categorize data, then split data to training and testing set
def setupDataset(filename: str , data_class: list):
    data = importData(filename)
    dataset, labels = manageData(data, data_class)
    training_data, testing_data, training_labels, testing_labels = trainTestSplit(dataset, labels)
    return training_data, testing_data, training_labels, testing_labels

