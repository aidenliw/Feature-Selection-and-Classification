import numpy as np

# Principle Component Analysis
class PCA:
    # Calculate the basic parameters of PCA
    # Return eigenvectors matrix and t score matrix
    def PCA_Params(data: np.ndarray):
        # Calculate and get the mean centered matrix.
        Xmc = data - np.mean(data)
        eigenvalues,eigenvectors = np.linalg.eig(np.dot(Xmc.T,Xmc))
        # Get the index of sorted value for the eigenvalues, from large to small
        sortIndex = np.argsort(eigenvalues)[::-1]
        # Sort the eigenvectors by the value of the eigenvalues
        sortedEigenvectors = eigenvectors[sortIndex]
        # Calculate the T scores/projection value of the data points
        t_scores = np.dot(Xmc,sortedEigenvectors)
        return sortedEigenvectors, t_scores


    # Calculate and get each value of Mean Squared Error(MSE) and Classification Error on dimensionality reduction
    # Inputs: Dataset, Class Labels, eigenvector matrix, T score matrix
    # Return: Mean Square Error list, Classification Error list
    def PCA_Debug(train_data: np.ndarray, train_label: np.ndarray, test_data: np.ndarray, test_label: np.ndarray, classifier):
        # Calculate and get the mean centered matrix.
        Xmc = train_data - np.mean(train_data)
        train_eigvec, train_tscores = PCA.PCA_Params(train_data)
        test_eigvec, test_tscores = PCA.PCA_Params(test_data)
        # Get how many components/features the dataset has
        dimension = len(train_data[0])
        # Initialize the empty arrays to store the corresponding data
        meanSquareError = np.zeros(dimension,)
        classificationError = np.zeros(dimension,)
        accuracyList = np.zeros(dimension,)

        # Format the class label array from vector column (n*1) to a single row list (1*n) 
        train_label = np.transpose(train_label)[0]
        test_label = np.transpose(test_label)[0]
        
        # Loop through data from max columns to 1 columns, and record the MSE and classification error
        for numDims in range(dimension,0,-1): 
            # reconstruction
            # Reduce the dimensionality from max, max-1, max-2 ... to 1
            train_reduced = train_tscores[:,0:numDims]
            eigvec_reduced = train_eigvec[:,0:numDims]
            test_reduced = test_tscores[:,0:numDims]
            # Reconstruct the original n-dimensional data with the shape of #rows x #columns
            XReconstructed = np.dot(train_reduced, np.transpose(eigvec_reduced))
            # Store the Mean Squared Error value of n dimentional data 
            # Mean squared error (MSE) measures the amount of error in statistical models
            meanSquareError[dimension-numDims] = sum(sum((XReconstructed - Xmc)**2))/len(XReconstructed)

            # Apply classification algorithm
            prediction, _, _ = classifier(train_reduced, test_reduced, train_label)
            # Determine the classification error
            errors = sum(prediction != test_label)
            accuracy = 1 - errors/len(test_label)
            accuracyList[dimension-numDims] = accuracy
            classificationError[dimension-numDims] = errors
            
        return meanSquareError, classificationError, accuracyList
    

    # After a best dimension value is found, apply PCA again to calculate the prediction value
    # Inputs: Dataset, Class Labels, Number of dimension to remove
    # Return: Test label, Predicted label, accuracy
    def PCA(train_data: np.ndarray, train_label: np.ndarray, test_data: np.ndarray, test_label: np.ndarray, numDims: int, classifier):
        train_eigvec, train_tscores = PCA.PCA_Params(train_data)
        test_eigvec, test_tscores = PCA.PCA_Params(test_data)

        # Get how many components/features the dataset has
        dimension = len(train_data[0])
        # Calculate how many components should be kept
        numDims = dimension - numDims

        # reconstruction
        # Reduce the dimensionality from input value
        train_reduced = train_tscores[:,0:numDims]
        eigvec_reduced = train_eigvec[:,0:numDims]
        test_reduced = test_tscores[:,0:numDims]
        
        # Format the class label array from vector column (n*1) to a single row list (1*n) 
        train_label = np.transpose(train_label)[0]
        test_label = np.transpose(test_label)[0]

        # Apply classification algorithm
        prediction, train_time, test_time = classifier(train_reduced, test_reduced, train_label)
        error = sum(prediction != test_label)
        accuracy = 1 - error/len(test_label)
        return test_label, prediction, accuracy, train_time, test_time