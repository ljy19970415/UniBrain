from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os

def plot_auc_pr(GT_list,pred_list,dis,rootdir, mode):
    # pred_list and GT_list are assumed to be numpy arrays or lists
    rootdir = rootdir+"curve_"+mode+'/'
    os.makedirs(rootdir,exist_ok=True)
    fpr, tpr, _ = roc_curve(GT_list, pred_list)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(GT_list, pred_list)
    pr_auc = auc(recall, precision)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve '+dis)
    plt.legend(loc="lower right")
    plt.savefig(rootdir+'roc_'+dis+'.png')
    # plt.show()

    # Plot PR curve
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve '+dis)
    plt.legend(loc="lower right")
    plt.savefig(rootdir+'pr_'+dis+'.png')
    # plt.show()