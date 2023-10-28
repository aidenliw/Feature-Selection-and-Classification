import numpy as np
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB

# Defind several clssification algorithms
class Classifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def applyClassification(self, training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
        # Define the model of the classifier
        lda = self.classifier
        # classification training
        lda.fit(training_data, class_labels)
        # classification testing
        predict_labels = lda.predict(testing_data)
        return predict_labels
    

    # def linearDiscriminantAnalysis(self, training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
    #     # Define the model of the classifier
    #     lda = LinearDiscriminantAnalysis()
    #     # classification training
    #     lda.fit(training_data, class_labels)
    #     # classification testing
    #     predict_labels = lda.predict(testing_data)
    #     return predict_labels

    # # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # def logisticRegression(self, training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
    #     # Define the model of the classifier
    #     lr = LogisticRegression()
    #     # classification training
    #     lr.fit(training_data, class_labels)
    #     # classification testing
    #     predict_labels = lr.predict(testing_data)
    #     return predict_labels
    
    # # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    # def gaussianNaiveBayes(self, training_data: np.ndarray, testing_data: np.ndarray, class_labels: np.ndarray):
    #     # Define the model of the classifier
    #     gnb = GaussianNB()
    #     # classification training
    #     gnb.fit(training_data, class_labels)
    #     # classification testing
    #     predict_labels = gnb.predict(testing_data)
    #     return predict_labels

