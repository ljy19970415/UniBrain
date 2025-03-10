from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import openpyxl as op

def op_toexcel(result, filename): # openpyxl库储存数据到excel
    # result fid:{"pred":"","gt":""}
    wb = op.Workbook() # 创建工作簿对象
    ws = wb['Sheet'] # 创建子表
    ws.append(['fid','gt','pred']) # 添加表头
    for fid in result:
        d = fid, result[fid]["gt"], result[fid]["pred"]
        ws.append(d) # 每次写入一行
    wb.save(filename)

def check_pred(target_class,fids,threshs,pred,gt,strict_test,filename):
    result = {fid:{"pred":[],"gt":[]} for fid in fids}
    for i in range(len(target_class)):
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].cpu().numpy()
        pred_label = (pred_np>=threshs[i])*1
        mask = (gt_np == -1).squeeze()
        if strict_test:
            gt_np[mask] = 0
            pred_label[mask] = 0
        else:
            gt_np[mask] = pred_label[mask]
        fid_pred = fids[pred_label==1]
        fid_real = fids[gt_np==1]

        for fid in fid_pred:
            result[fid]["pred"].append(target_class[i])
        for fid in fid_real:
            result[fid]["gt"].append(target_class[i])
    for fid in result:
        result[fid]["pred"].sort()
        result[fid]["gt"].sort()
        result[fid]["pred"] = ','.join(result[fid]["pred"])
        result[fid]["gt"] = ",".join(result[fid]["gt"])
    op_toexcel(result, filename)

def check_pred_new(target_class,fids,threshs,pred,gt,strict_test,filename):
    result = {fid:{"pred":[],"gt":[]} for fid in fids}
    for i in range(len(target_class)):
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].cpu().numpy()
        pred_label = (pred_np>=threshs[i])*1
        # mask = (gt_np == -1).squeeze()
        # if strict_test:
        #     gt_np[mask] = 0
        #     pred_label[mask] = 0
        # else:
        #     gt_np[mask] = pred_label[mask]
        fid_pred = fids[pred_label==1]
        fid_real = fids[gt_np==1]

        for fid in fid_pred:
            result[fid]["pred"].append(target_class[i])
        for fid in fid_real:
            result[fid]["gt"].append(target_class[i])

    for fid in result:
        result[fid]["pred"].sort()
        result[fid]["gt"].sort()
        result[fid]["pred"] = ','.join(result[fid]["pred"])
        result[fid]["gt"] = ",".join(result[fid]["gt"])
    op_toexcel(result, filename)

def plot_auc_pr(GT_list,pred_list,dis,rootdir, mode):
    # pred_list and GT_list are assumed to be numpy arrays or lists
    rootdir = rootdir+"curve_"+mode+'/'
    os.makedirs(rootdir,exist_ok=True)
    fpr, tpr, _ = roc_curve(GT_list, pred_list)
    roc_auc = auc(fpr, tpr)
    index = 0
    for idx,i in enumerate(fpr):
        if i >= 0.1:
            index = idx
            break
    print(dis,"fp",fpr[index],tpr[index])

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