import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_cm(tag='train', y_true=None, y_pred=None, display_labels=None, save_dir=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    cmp.plot(ax=ax)
    # plt.show()
    plt.savefig(os.path.join(save_dir, tag + '_confusion_matrix.png'))
    txt_path = os.path.join(save_dir, tag + '_metric.txt')
    n_classes = cm.shape[0]
    for i in range(n_classes):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        metric_txt = f"Class {i}: Sensitivity = {tpr:.4f}, specificity = {tnr:.4f}\n"
        with open(txt_path, 'a') as f:
            f.write(metric_txt)
