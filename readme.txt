# Feature Selection and Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Solution](#solution)
4. [Discussion and Limitation](#discussion-and-limitation)
5. [Conclusion](#conclusion)
6. [Instructions to Execute the Code](#instructions-to-execute-the-code)
7. [References](#references)

## Introduction
This project aims to compare feature extraction approaches for the classification of rice species. The focus is on applying Principal Component Analysis (PCA) and Backward Search to a dataset of rice grain images. The classifiers used are Gaussian Naïve Bayes and Logistic Regression to classify the rice grains into two species: Cammeo and Osmancik. The objectives include evaluating the performance of these classifiers, comparing their results through confusion matrices, and recording the computational time for training and testing processes.

## Problem Statement
The dataset consists of 3810 rice grain images belonging to two species: Cammeo and Osmancik. Seven morphological features were extracted for each grain:
- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- Extent

The dataset can be retrieved from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik).

## Solution

### Applying PCA with Gaussian Naïve Bayes
PCA was applied to reduce the dimensionality of the feature set while retaining most of the variance. The reduced feature set was used to train a Gaussian Naïve Bayes classifier.

- **Feature extraction time:** 50.76 milliseconds
- **Training time:** 1.45 milliseconds
- **Testing time:** 0.39 milliseconds
- **Dimension Reduction:** Removed 2 dimensions
- **Accuracy:** 90.64%

#### Confusion Matrix
- **Cammeo correctly classified (True positive):** 447 (86.29%)
- **Cammeo misclassified as Osmancik (False negative):** 71 (13.71%)
- **Osmancik misclassified as Cammeo (False positive):** 36 (5.76%)
- **Osmancik correctly classified (True negative):** 589 (94.24%)

### Applying PCA with Logistic Regression
PCA was applied to extract important features, and the reduced feature set was used to train a Logistic Regression classifier.

- **Feature extraction time:** 255.14 milliseconds
- **Training time:** 69.04 milliseconds
- **Testing time:** 0.21 milliseconds
- **Dimension Reduction:** Removed 2 dimensions
- **Accuracy:** 93.26%

#### Confusion Matrix
- **Cammeo correctly classified (True positive):** 473 (91.31%)
- **Cammeo misclassified as Osmancik (False negative):** 45 (8.69%)
- **Osmancik misclassified as Cammeo (False positive):** 32 (5.12%)
- **Osmancik correctly classified (True negative):** 593 (94.88%)

### Applying Backward Search with Gaussian Naïve Bayes
Backward Search systematically eliminates less informative features, and the remaining features were used to train a Gaussian Naïve Bayes classifier.

- **Feature extraction time:** 197.53 milliseconds
- **Training time:** 1.38 milliseconds
- **Testing time:** 0.35 milliseconds
- **Feature Selection:** Removed 5 dimensions
- **Accuracy:** 92.04%

#### Confusion Matrix
- **Cammeo correctly classified (True positive):** 482 (93.05%)
- **Cammeo misclassified as Osmancik (False negative):** 36 (6.95%)
- **Osmancik misclassified as Cammeo (False positive):** 55 (8.80%)
- **Osmancik correctly classified (True negative):** 570 (91.20%)

### Applying Backward Search with Logistic Regression
Backward Search was used to select the most informative features, and the selected features were employed to train a Logistic Regression classifier.

- **Feature extraction time:** 857.30 milliseconds
- **Training time:** 13.97 milliseconds
- **Testing time:** 0.24 milliseconds
- **Feature Selection:** Removed 2 dimensions
- **Accuracy:** 93.18%

#### Confusion Matrix
- **Cammeo correctly classified (True positive):** 479 (92.47%)
- **Cammeo misclassified as Osmancik (False negative):** 39 (7.53%)
- **Osmancik misclassified as Cammeo (False positive):** 39 (6.24%)
- **Osmancik correctly classified (True negative):** 586 (93.76%)

## Discussion and Limitation
In this project, we explored four different combinations of feature extraction methods and classifiers for the classification of rice species based on morphological features. PCA with Logistic Regression achieved the highest accuracy of 93.26% after removing 2 dimensions. However, it requires more computational time than many other combinations. PCA with Gaussian Naïve Bayes offers good accuracy with shorter computational time.

## Conclusion
The choice of the optimal combination should align with the project's requirements and constraints. PCA with Logistic Regression is recommended for achieving the highest accuracy, while PCA with Gaussian Naïve Bayes is suitable for scenarios where computational time is critical. 

## Instructions to Execute the Code

**To execute the code:**
1. **Python Script:**
   - Open `main.py` with any Python compiler such as Visual Studio Code or PyCharm.
   - Run the `main.py` file without debug mode.
   - The results will be displayed on the terminal screen.

2. **Jupyter Notebook:**
   - Open the Jupyter Notebook file (`file.ipynb`) in Google Colab or Jupyter Notebook.
   - Execute the cells to see the results.

## References
- [Rice Cammeo and Osmancik Dataset](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)
