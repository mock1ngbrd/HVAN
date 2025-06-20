import time
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
import logging
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import os

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
        return 0  

    accuracy = correct / total
    return accuracy

def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def test_final(model, weight_path, test_dataloader, mod='both'):
    model.module.load_state_dict(torch.load(weight_path))
    model.eval()
    val_y_true = []
    val_y_pred_sm = []

    for idx, val_batch in enumerate(test_dataloader):
        start_time = time.perf_counter()
        val_bmode = val_batch['volume1'].float().cuda()
        val_swe = val_batch['volume2'].float().cuda()
        val_label = val_batch['cspca']
        val_name = val_batch['name']
        with torch.no_grad():
            if mod is 'both':
                y = model(val_bmode, val_swe)
            elif mod is 'swe':
                y = model(val_swe)
            elif mod is 'bmode':
                y = model(val_bmode)
            else:
                return
        if isinstance(y, list):
            y = y[0]
        y_sm = F.softmax(y, dim=1)
        val_y_true.extend(val_label)
        val_y_pred_sm.extend(y_sm[:, 1])
        end_time = time.perf_counter()
        execution_time_seconds = end_time - start_time
        logging.info("test index: {:2d}/{:2d}  case id: {}   label: {}   pred: {}  time:{:.4f}".
                     format(idx, len(test_dataloader), val_name, val_label.cpu(), torch.max(y, 1)[1].cpu(), execution_time_seconds))

    val_y_true = torch.stack(val_y_true, dim=0)
    val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
    fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
    precision, recall, thresholds = precision_recall_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy())
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    pred_list = [1 if x > best_threshold else 0 for x in val_y_pred_sm.cpu().data.numpy()]
    acc = accuracy_score(val_y_true.cpu().data.numpy(), pred_list)
    val_auc = auc(fpr, tpr)
    sensitivity, specificity = calculate_sensitivity_specificity(val_y_true.cpu().data.numpy(), pred_list)
    pos_acc = compute_accuracy_by_class(val_y_true, pred_list, 1)
    neg_acc = compute_accuracy_by_class(val_y_true, pred_list, 0)
    logging.info("best model in test result: AUC=%.4f Sen=%.4f Spe=%.4f f1_score=%.4f ACC=%.4f"
                 % (val_auc, sensitivity, specificity,
                    f1_scores[best_threshold_index], acc))
    logging.info('positive acc: {:.4f} negative acc: {:.4f}'.format(pos_acc, neg_acc))

