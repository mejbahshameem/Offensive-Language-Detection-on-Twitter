import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np

# for calculating different scores(accuracy, recall, precision, f1)
def calculate_measures(classifier, testClass, testGuess, title, plot_confusion=True):

    # Compare the accuracy of the output (Yguess) with the class labels of the dev/test set (Ytest)
    print("Accuracy = "+str(accuracy_score(testClass, testGuess)))
    print("F1-score(macro) = "+str(f1_score(testClass, testGuess, average='macro')))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the dev/test set (Ytest)
    print(classification_report(testClass, testGuess, labels=classifier.classes_, target_names=None, sample_weight=None, digits=3))

    # prints the confution matrix in terminal
    print_confusion_matrix(testClass, testGuess, classes=classifier.classes_)

    # Drawing Confusion Matrix
    if plot_confusion:
        plot_confusion_matrix(testClass, testGuess, classes=classifier.classes_, normalize=True, title=title)
        plt.savefig('Confusion_Matrix.png')
        # plt.show()

#Prints the confusion Matrix in Terminal
def print_confusion_matrix(testClass, testGuess, classes):


    # Showing the Confusion Matrix
    print("Confusion Matrix (class):")
    cm = confusion_matrix(testClass, testGuess, labels=classes)
    print(classes)
    print(cm)
    print()

    # Compute confusion matrix of Accuracy
    print("Normalized confusion matrix (Accuracy)")
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(classes)
    print(cm)

# This function Plots the confusion matrix (as image)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix (Accuracy)'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# Draw plots based on different values of the parameter C and save the plot
def draw_plots(val, accu, f1, value_name):
    plt.plot(val, accu, color='red', label='Accuracy')
    plt.plot(val, f1, color='yellow', label='F1-score')

    x_label = "Values of "+value_name
    plt.xlabel(x_label)
    plt.ylabel('Scores')
    plt.legend()

    plt.savefig('Linear_SVM_Plot.png')