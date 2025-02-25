import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score, classification_report, precision_score, f1_score, accuracy_score

def get_optimal_threshold(y_true, decision_scores):
    fpr, tpr, thresholds = roc_curve(y_true, decision_scores)
    # 下方樣本準確-上方樣本準確，最大值為最佳閾值，即youden index
    # 這樣的閾值可以使得偽陽性率最低，真陽性率最高
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, fpr, tpr, thresholds

def get_score(y_true, decision_scores, threshold=None):
    if threshold is None:
        threshold, fpr, tpr, thresholds = get_optimal_threshold(y_true, decision_scores)
        # plot_roc_curve(fpr, tpr, thresholds, optimal_threshold=threshold)
    
    y_pred = (decision_scores >= threshold).astype(int)
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    classification = classification_report(y_true, y_pred, digits=4)
    
    return conf_matrix, recall, precision, f1, accuracy, classification, threshold

def plot_roc_curve(fpr, tpr, thresholds, optimal_threshold):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], marker='o', color='red', label='Optimal Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def calculate_recall_precision(confusion_matrix):
    """
    Calculate recall and precision from a given confusion matrix.
    
    Parameters:
        confusion_matrix (numpy.ndarray): A 2x2 confusion matrix.
                                         [[TP, FP],
                                          [FN, TN]]
    
    Returns:
        recall (float): Recall value.
        precision (float): Precision value.
    """
    # Extract values from the confusion matrix
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TN = confusion_matrix[0, 0]

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = (2 * precision * recall) / (precision + recall)
    acc = (TP + TN) / (TP + FP + FN + TN)

    return recall, precision, f1_score, acc