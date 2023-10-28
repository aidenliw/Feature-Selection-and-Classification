import numpy as np

class FeatureSelection:
    # Use feature selection with backward search for a N dimensional dataset
    def backwardSearchDebug(train_data: np.ndarray, train_label: np.ndarray, test_data: np.ndarray, test_label: np.ndarray, classifier):

        dimension = len(train_data[0])
        # Store the Classification errors
        classificationError = 10000*np.ones(dimension)
        # Format the class label array from vector column (n*1) to a single row list (1*n) 
        train_label = np.transpose(train_label)[0]
        test_label = np.transpose(test_label)[0]
        # Train and test the whole data, store the classification error into list
        prediction = classifier(train_data, test_data, train_label)

        error = sum(prediction != test_label)
        classificationError[0] = error
        accuracyList = np.zeros(dimension,)
        accuracy = 1 - error/len(test_label)
        accuracyList[0] = accuracy
        indexsToRemove = []    

        # Iterate over all selected features
        for iteration in range(1, dimension):
            # Store the column index that should be removed for the min error dataset 
            minErrorIndex = 0
            # Iterate over all columns of the dataset from column index n to 0
            # Find the index of column that has the worst feature
            for i in range(dimension):
                
                # Remove the i-th column from the dataset for each iteration     
                train_reduced = np.delete(train_data, i, 1)
                test_reduced = np.delete(test_data, i, 1)
                # Classify the training data using currently selected features
                prediction = classifier(train_reduced, test_reduced, train_label)
                error = sum(prediction != test_label)
                # Check whether the dataset of this batch has the minimum error value
                if error < classificationError[iteration]:
                    classificationError[iteration] = error
                    minErrorIndex = i
                accuracy = 1 - error/len(test_label)
                accuracyList[iteration] = accuracy
            # Remove the feature column, then update the datasets
            indexsToRemove.append(minErrorIndex)
            train_data = np.delete(train_data, minErrorIndex, 1)
            test_data = np.delete(test_data, minErrorIndex, 1)
            dimension = dimension - 1
        return classificationError, accuracyList, indexsToRemove
    
    
    # After a best dimension value is found, apply Backward Search again to calculate the prediction value
    def backwardSearch(train_data: np.ndarray, train_label: np.ndarray, test_data: np.ndarray, test_label: np.ndarray, indexsToRemove: list, classifier):

        dimension = len(train_data[0])
        # Store the Classification errors
        classificationError = 10000*np.ones(dimension)
        # Format the class label array from vector column (n*1) to a single row list (1*n) 
        train_label = np.transpose(train_label)[0]
        test_label = np.transpose(test_label)[0]
        # Train and test the whole data, store the classification error into list
        prediction = classifier(train_data, test_data, train_label)

        error = sum(prediction != test_label)
        classificationError[0] = error
        accuracyList = np.zeros(dimension,)
        accuracy = 1 - error/len(test_label)
        accuracyList[0] = accuracy

        # Iterate the indexs that needed to be removed
        for index in indexsToRemove:
            train_data = np.delete(train_data, index, 1)
            test_data = np.delete(test_data, index, 1)

        prediction = classifier(train_data, test_data, train_label)
        error = sum(prediction != test_label)
        accuracy = 1 - error/len(test_label)
        return test_label, prediction, accuracy