import matplotlib.pyplot as plt

def draw_roc_curve(true_labels, predicted_probabilities):
    fpr, tpr, thresholds = [], [], []
    for threshold in range(101):
        threshold /= 100
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(true_labels)):
            if predicted_probabilities[i] >= threshold:
                if true_labels[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if true_labels[i] == 1:
                    fn += 1
                else:
                    tn += 1
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
        thresholds.append(threshold)
    auc = 0
    for i in range(100):
        auc += -((fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) / 2)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Area Under the ROC Curve:', auc)
    
    
import pandas as pd
# Import true labels from CSV file
true_labels = pd.read_csv('hw06_true_labels.csv')['1']

# Import predicted probabilities from CSV file
predicted_probabilities = pd.read_csv('hw06_predicted_probabilities.csv')['0.639676']

print(draw_roc_curve(true_labels, predicted_probabilities))


def draw_pr_curve(true_labels, predicted_probabilities):
    precision, recall, thresholds = [], [], []
    for threshold in range(101):
        threshold /= 100
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(true_labels)):
            if predicted_probabilities[i] >= threshold:
                if true_labels[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if true_labels[i] == 1:
                    fn += 1
                else:
                    tn += 1
        precision.append(tp / (tp + fp + 1e-9))
        recall.append(tp / (tp + fn + 1e-9))
        thresholds.append(threshold)
    auc = 0
    for i in range(100):
        auc += -((recall[i + 1] - recall[i]) * (precision[i] + precision[i + 1]) / 2)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.show()
    print('Area Under the PR Curve:', auc)

print(draw_pr_curve(true_labels, predicted_probabilities))

