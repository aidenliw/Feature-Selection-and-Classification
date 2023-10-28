
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Plot:

    # Plot the data of Error 
    def plotErrors(data1: list, data2: list, data3: list, range: int, 
                title1: str, title2: str, title3: str, xlabel: str, 
                ylabel1: str, ylabel2: str, ylabel3: str, subtitle: str):
        n = np.linspace(0,range-1,range)
        fig_sim, ax_sim = plt.subplots(1, 3, figsize=(16, 4))
        ax_sim[0].plot(n, data1[0:range])
        ax_sim[0].set_xlabel(xlabel)
        ax_sim[0].set_ylabel(ylabel1)
        ax_sim[0].set_title(title1)
        
        ax_sim[1].plot(n, data2[0:range])
        ax_sim[1].set_xlabel(xlabel)
        ax_sim[1].set_ylabel(ylabel2)
        ax_sim[1].set_title(title2)

        ax_sim[2].plot(n, data3[0:range])
        ax_sim[2].set_xlabel(xlabel)
        ax_sim[2].set_ylabel(ylabel3)
        ax_sim[2].set_title(title3)
        plt.suptitle(subtitle)
        plt.show()


    # Plot the data of Error 
    def plotErrors2(data1: list, data2: list, range: int, 
                title1: str, title2: str, xlabel: str, 
                ylabel1: str, ylabel2: str, subtitle: str):
        n = np.linspace(0,range-1,range)
        fig_sim, ax_sim = plt.subplots(1, 2, figsize=(12, 4))
        ax_sim[0].plot(n, data1[0:range])
        ax_sim[0].set_xlabel(xlabel)
        ax_sim[0].set_ylabel(ylabel1)
        ax_sim[0].set_title(title1)
        
        ax_sim[1].plot(n, data2[0:range])
        ax_sim[1].set_xlabel(xlabel)
        ax_sim[1].set_ylabel(ylabel2)
        ax_sim[1].set_title(title2)
        plt.suptitle(subtitle)
        plt.show()


    # # Plot the comparison for 2 datasets 
    # def plotComparison(data1: list, data2: list, range: int, title: str, xlabel: str, ylabel: str, xName: str, yName: str):
    #     n = np.linspace(0,range-1,range)
    #     # Plot the variance data for comparision
    #     plt.plot(n,data1[0:range], label=xName)
    #     plt.plot(n,data2[0:range], label=yName)
    #     plt.legend(loc="upper left")
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.show()

    
    # Plot confusion matrix by using sklearn
    # Inputs: Confusion matrix ndarray
    def plotConfusionMatrix(cm, data_class):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_class)
        disp.plot()
        plt.show()

    
    # Calculate the True Positive, False Negative values for each attribute/feature
    # Print the data
    def calculateConfusionMatrixValue(cm, data_class):
        # Calculate the values for the whole matrix, store each value into arrays
        true_positives = np.diag(cm)
        false_negatives = cm.sum(axis=1) - np.diag(cm)
        false_positives = cm.sum(axis=0) - np.diag(cm)  
        true_negatives = cm.sum() - (false_positives + false_negatives + true_positives)
        true_positive_rates = true_positives/(true_positives+false_negatives)
        false_negative_rates = false_negatives/(true_positives+false_negatives)
        false_positive_rates = false_positives/(false_positives+true_negatives)
        true_negative_rates = true_negatives/(true_negatives+false_positives)

        print("{:<10} {:<15} {:<16} {:<16} {:<16}".format("Class", "True Positives", "False Negatives", "False Positives", "True Negatives"))
        for i in range(len(cm)):
            print("{:<10} {:<15} {:<16} {:<16} {:<16}".format(
                data_class[i],
                "%d (%0.2f%%)" % (true_positives[i], 100*true_positive_rates[i]),
                "%d (%0.2f%%)" % (false_negatives[i], 100*false_negative_rates[i]),
                "%d (%0.2f%%)" % (false_positives[i], 100*false_positive_rates[i]),
                "%d (%0.2f%%)" % (true_negatives[i], 100*true_negative_rates[i]),
            ))
