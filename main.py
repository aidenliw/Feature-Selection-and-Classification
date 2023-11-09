# Author: Aiden WangYang Li 
import numpy as np
np.set_printoptions(suppress=True) # setting the printing options 
from sklearn.metrics import confusion_matrix
import time
import DataLoader as dl
from Classifiers import *
from PCA import PCA
from FeatureSelection import FeatureSelection
from Plot import Plot


# Set up class labels list
data_class = ["Cammeo", "Osmancik"]
# Input the number of features in dataset
num_features = 7
# Read data from the provided path
datafile = "data/rice_data.csv"

training_data, testing_data, training_labels, testing_labels = dl.setupDataset(datafile, data_class)
# print("\nTraining Set:")
# print(np.concatenate((training_data, training_labels), axis=1))

cls1 = Classifier(GaussianNB())
cls2 = Classifier(LogisticRegression())

''' Apply PCA with Naive Bayes. Iteration on dimensionality reduction to find the best condition '''
print("\nApplying PCA with Naive Bayes")
start = time.time()
meanSquareError, classificationError, accuracyList \
    = PCA.PCA_Debug(training_data, training_labels, testing_data, testing_labels, cls1.applyClassification)
end = time.time()
print(" > Computational Times for finding best dimension is %0.2f milliseconds." % ((end - start) * 1000))

# Plot the line chart for mean square error, classification error, and accuracy list
Plot.plotErrors(meanSquareError, classificationError, accuracyList, num_features, 
           "Mean Squared Errors", "Classification Errors", "Accuracy Trend",
           "Number of Dimensions Removed",
           "Mean Squared Error", "Number of Errors", "Accuracy",
           'Principal Component Analysis using Naive Bayes')

# By reviewing above plots, removing 2 dimensions is the best decision
dim_remove1 = 2
true_labels, pred_labels, accuracy, train_time, test_time \
    = PCA.PCA(training_data, training_labels, testing_data, testing_labels, dim_remove1, cls1.applyClassification)
print(" > Accuracy: %.2f%%" % (accuracy*100))
print(" > Computational Times for training data is %0.2f milliseconds." % (train_time * 1000))
print(" > Computational Times for testing data is %0.2f milliseconds." % (test_time * 1000))

# Calculate the confusion matrix
print(" > Confusion Matrix:")
cm1 = confusion_matrix(true_labels, pred_labels)
Plot.plotConfusionMatrix(cm1, data_class)
Plot.calculateConfusionMatrixValue(cm1, data_class)


''' Apply PCA with Logistic Regression. Iteration on dimensionality reduction to find the best condition '''
print("\nApplying PCA with Logistic Regression")
start = time.time()
meanSquareError, classificationError, accuracyList \
    = PCA.PCA_Debug(training_data, training_labels, testing_data, testing_labels, cls2.applyClassification)
end = time.time()
print(" > Computational Times for finding best dimension is %0.2f milliseconds." % ((end - start) * 1000))

# Plot the line chart for mean square error, classification error, and accuracy list
Plot.plotErrors(meanSquareError, classificationError, accuracyList, num_features, 
           "Mean Squared Errors", "Classification Errors", "Accuracy Trend",
           "Number of Dimensions Removed",
           "Mean Squared Error", "Number of Errors", "Accuracy",
           'Principal Component Analysis using Logistic Regression')

# By reviewing above plots, removing 2 dimensions is the best decision
dim_remove2 = 2
true_labels, pred_labels, accuracy, train_time, test_time \
    = PCA.PCA(training_data, training_labels, testing_data, testing_labels, dim_remove2, cls2.applyClassification)
print(" > Accuracy: %.2f%%" % (accuracy*100))
print(" > Computational Times for training data is %0.2f milliseconds." % (train_time * 1000))
print(" > Computational Times for testing data is %0.2f milliseconds." % (test_time * 1000))

# Calculate the confusion matrix by using sklearn
print(" > Confusion Matrix:")
cm2 = confusion_matrix(true_labels, pred_labels)
Plot.plotConfusionMatrix(cm2, data_class)
Plot.calculateConfusionMatrixValue(cm2, data_class)


''' Apply Backward Search with Naive Bayes. Iteration on dimensionality reduction to find the best condition '''
print("\nApplying Backward Search with Naive Bayes")
start = time.time()
classificationError, accuracyList, indexsToRemove1 \
    = FeatureSelection.backwardSearchDebug(training_data, training_labels, testing_data, testing_labels, cls1.applyClassification)
end = time.time()
print(" > Computational Times for finding best dimension is %0.2f milliseconds." % ((end - start) * 1000))

# Plot the line chart for classification error, and accuracy list
Plot.plotErrors2(classificationError, accuracyList, num_features, 
           "Classification Errors", "Accuracy Trend",
           "Number of Dimensions Removed",
           "Number of Errors", "Accuracy",
           'Backward Search using Naive Bayes')

# By reviewing above plots, removing 5 dimensions is the best decision
dim_remove3 = 5
true_labels1, pred_labels1, accuracy, train_time, test_time \
    = FeatureSelection.backwardSearch(training_data, training_labels, testing_data, testing_labels, indexsToRemove1[0:dim_remove3], cls1.applyClassification)
print(" > Accuracy: %.2f%%" % (accuracy*100))
print(" > Computational Times for training data is %0.2f milliseconds." % (train_time * 1000))
print(" > Computational Times for testing data is %0.2f milliseconds." % (test_time * 1000))

# Calculate the confusion matrix by using sklearn
print(" > Confusion Matrix:")
cm3 = confusion_matrix(true_labels1, pred_labels1)
Plot.plotConfusionMatrix(cm3, data_class)
Plot.calculateConfusionMatrixValue(cm3, data_class)


''' Apply Backward Search with Logistic Regression. Iteration on dimensionality reduction to find the best condition '''
print("\nApplying Backward Search with Logistic Regression")
start = time.time()
classificationError, accuracyList, indexsToRemove2 \
    = FeatureSelection.backwardSearchDebug(training_data, training_labels, testing_data, testing_labels, cls2.applyClassification)
end = time.time()
print(" > Computational Times for finding best dimension is %0.2f milliseconds." % ((end - start) * 1000))

Plot.plotErrors2(classificationError, accuracyList, num_features, 
           "Classification Errors", "Accuracy Trend",
           "Number of Dimensions Removed",
           "Number of Errors", "Accuracy",
           'Backward Search using Logistic Regression')

# By reviewing above plots, removing 2 dimensions is the best decision
dim_remove4 = 2
true_labels2, pred_labels2, accuracy, train_time, test_time \
    = FeatureSelection.backwardSearch(training_data, training_labels, testing_data, testing_labels, indexsToRemove2[0:dim_remove4], cls2.applyClassification)
print(" > Accuracy: %.2f%%" % (accuracy*100))
print(" > Computational Times for training data is %0.2f milliseconds." % (train_time * 1000))
print(" > Computational Times for testing data is %0.2f milliseconds." % (test_time * 1000))

# Calculate the confusion matrix by using sklearn
print(" > Confusion Matrix:")
cm4 = confusion_matrix(true_labels2, pred_labels2)
Plot.plotConfusionMatrix(cm4, data_class)
Plot.calculateConfusionMatrixValue(cm4, data_class)
