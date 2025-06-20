import time

import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
import logging
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import math

import warnings

warnings.filterwarnings('ignore')


def compute_accuracy_by_class(ground_truth, predicted, class_label):
    correct = 0
    total = 0

    for true_label, predicted_label in zip(ground_truth, predicted):
        if true_label == class_label:
            total += 1
            if predicted_label == class_label:
                correct += 1

    if total == 0:
        return 0  # 避免除零错误

    accuracy = correct / total
    return accuracy


def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def cal_f1(label_list, score):
    thr = 0.5
    pred = [1 if x >= thr else 0 for x in score]
    f1_scores = f1_score(label_list, pred)
    return f1_scores

def cal_metrics(val_y_true, val_y_pred_sm):
    fpr, tpr, _ = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
    f1_scores = cal_f1(val_y_true, val_y_pred_sm)
    pred_list = [1 if x > 0.5 else 0 for x in val_y_pred_sm.cpu().data.numpy()]
    acc = accuracy_score(val_y_true.cpu().data.numpy(), pred_list)
    val_auc = auc(fpr, tpr)
    sensitivity, specificity = calculate_sensitivity_specificity(val_y_true.cpu().data.numpy(), pred_list)
    res = [val_auc, f1_scores, acc, sensitivity, specificity]
    data = np.array(res)
    data[np.isnan(data)] = 0
    return data.tolist()

def crossval(model, val_dataloader, writer, epoch, exp_save_path,
                           best_auc, k_index, mod=True):
    model.eval()
    val_y_true = []
    val_y_pred = []
    # val_2d_true = []
    # val_2d_pred = []
    for idx, val_batch in enumerate(tqdm(val_dataloader)):
        # start_time = time.perf_counter()
        # val_bmode = val_batch['volume1'].float().cuda()
        # val_swe = val_batch['volume2'].float().cuda()
        val_label = val_batch['cspca']
        # val_name = val_batch['name'][0]
        with torch.no_grad():
            if mod == 'both':
                val_bmode = val_batch['volume1'].float().cuda()
                val_swe = val_batch['volume2'].float().cuda()
                y = model(val_bmode, val_swe)
            elif mod == 'single':
                val_swe = val_batch['volume'].float().cuda()
                y = model(val_swe)
            else:
                raise -1

        if isinstance(y, list):
            out, out2d = y[0], y[1]
        else:
            out = y
        out_sg = F.sigmoid(out)
        val_y_true.extend(val_label)
        val_y_pred.extend(out_sg)
        # end_time = time.perf_counter()
        # execution_time_seconds = end_time - start_time

    val_y_true = torch.stack(val_y_true, dim=0)
    val_y_pred = torch.stack(val_y_pred, dim=0)
    metrics3d = cal_metrics(val_y_true, val_y_pred)

    logging.info("fold:%d val 3d result(%d): \nAUC=%.4f f1_score=%.4f ACC=%.4f Sen=%.4f Spe=%.4f"
                 % (k_index, epoch, metrics3d[0], metrics3d[1], metrics3d[2],
                    metrics3d[3], metrics3d[4]))

    writer.add_scalar(f"val{k_index}/auc3d", metrics3d[0], global_step=epoch)
    writer.add_scalar(f"val{k_index}/acc", metrics3d[2], global_step=epoch)
    val_auc3d = metrics3d[0]
    if val_auc3d >= best_auc[k_index, 0]:
        best_auc[k_index] = np.array(metrics3d)
        torch.save(model.module.state_dict(), '{}/ckp_model/modelbest_fold{}.pth'.format(exp_save_path, k_index))

    writer.add_scalar(f"val{k_index}/best_model_result_auc3d", best_auc[k_index, 0], global_step=epoch)
    return best_auc


def crossval_last(model, val_dataloader,
                           best_auc, k_index, mod=True):
    model.eval()
    val_y_true = []
    val_y_pred = []
    for idx, val_batch in enumerate(tqdm(val_dataloader)):
        val_label = val_batch['cspca']
        with torch.no_grad():
            if mod == 'both':
                val_bmode = val_batch['volume1'].float().cuda()
                val_swe = val_batch['volume2'].float().cuda()
                y = model(val_bmode, val_swe)
            elif mod == 'single':
                val_swe = val_batch['volume'].float().cuda()
                y = model(val_swe)
            else:
                raise -1

        if isinstance(y, list):
            out, out2d = y[0], y[1]
        else:
            out = y
        out_sg = F.sigmoid(out)
        val_y_true.extend(val_label)
        val_y_pred.extend(out_sg)

    val_y_true = torch.stack(val_y_true, dim=0)
    val_y_pred = torch.stack(val_y_pred, dim=0)
    metrics3d = cal_metrics(val_y_true, val_y_pred)

    best_auc[k_index] = np.array(metrics3d)

    return best_auc

def test(model, val_dataloader, best_auc, mod=True):
    model.eval()
    val_y_true = []
    # val_y_pred_sm = []
    val_y_pred = []
    # val_2d_true = []
    # val_2d_pred = []
    for idx, val_batch in enumerate(tqdm(val_dataloader)):
        # start_time = time.perf_counter()
        # val_bmode = val_batch['volume1'].float().cuda()
        # val_swe = val_batch['volume2'].float().cuda()
        val_label = val_batch['cspca']
        # val_name = val_batch['name'][0]
        with torch.no_grad():
            if mod == 'both':
                val_bmode = val_batch['volume1'].float().cuda()
                val_swe = val_batch['volume2'].float().cuda()
                y = model(val_bmode, val_swe)
            elif mod == 'single':
                val_swe = val_batch['volume'].float().cuda()
                y = model(val_swe)
            else:
                raise -1

        if isinstance(y, list):
            out, out2d = y[0], y[1]
        else:
            out = y
        # out_sm = F.softmax(out, dim=1)
        # val_y_true.extend(val_label)
        # val_y_pred_sm.extend(out_sm[:, 1])
        out_sg = F.sigmoid(out)
        val_y_true.extend(val_label)
        val_y_pred.extend(out_sg)
        # end_time = time.perf_counter()
        # execution_time_seconds = end_time - start_time

    val_y_true = torch.stack(val_y_true, dim=0)
    val_y_pred = torch.stack(val_y_pred, dim=0)
    metrics3d = cal_metrics(val_y_true, val_y_pred)

    logging.info("test result: \nAUC=%.4f f1_score=%.4f ACC=%.4f Sen=%.4f Spe=%.4f Thr=%.4f"
                 % (metrics3d[0], metrics3d[1], metrics3d[2],
                    metrics3d[3], metrics3d[4], metrics3d[7]))
    logging.info('positive acc: {:.4f} negative acc: {:.4f}'.format(metrics3d[5], metrics3d[6]))
    return best_auc
