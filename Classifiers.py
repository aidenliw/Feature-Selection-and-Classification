import numpy as np
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
import time

# Defind several clssification algorithms
class Classifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def applyClassification(self, training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
        # Define the model of the classifier
        lda = self.classifier
        # classification training
        train_start = time.time()
        lda.fit(training_data, class_labels)
        train_time = time.time() - train_start
        # classification testing
        test_start = time.time()
        predict_labels = lda.predict(testing_data)
        test_time = time.time() - test_start
        return predict_labels, train_time, test_time
    
    # # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
    # def linearDiscriminantAnalysis(training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
    #     # Define the model of the classifier
    #     lda = LinearDiscriminantAnalysis()

    #     # classification training
    #     train_start = time.time()
    #     lda.fit(training_data, class_labels)
    #     train_time = time.time() - train_start
    #     # classification testing
    #     test_start = time.time()
    #     predict_labels = lda.predict(testing_data)
    #     test_time = time.time() - test_start
    #     return predict_labels, train_time, test_time

    # # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # def logisticRegression(training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
    #     # Define the model of the classifier
    #     lr = LogisticRegression()
    #     # classification training
    #     train_start = time.time()
    #     lr.fit(training_data, class_labels)
    #     train_time = time.time() - train_start
    #     # classification testing
    #     test_start = time.time()
    #     predict_labels = lr.predict(testing_data)
    #     test_time = time.time() - test_start
    #     return predict_labels, train_time, test_time

    # # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    # def gaussianNaiveBayes(training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
    #     # Define the model of the classifier
    #     gnb = GaussianNB()
    #     # classification training
    #     train_start = time.time()
    #     gnb.fit(training_data, class_labels)
    #     train_time = time.time() - train_start
    #     # classification testing
    #     test_start = time.time()
    #     predict_labels = gnb.predict(testing_data)
    #     test_time = time.time() - test_start
    #     return predict_labels, train_time, test_time

