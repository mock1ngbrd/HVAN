import codecs
import os
import random

def save_train_val_test_txt(data_split, save_path, k_idx=0):

    train_list = data_split[0]
    val_list = data_split[1]
    if len(data_split) == 3:
        test_list = data_split[2]
    with codecs.open('{}/fold{}.txt'.format(save_path, k_idx), encoding='utf-8', mode='w') as f:
        f.write('train:\n')
        for train in train_list:
            f.write('%s\n' % train)
        f.write('val:\n')
        for val in val_list:
            f.write('%s\n' % val)
        if len(data_split) == 3:
            f.write('test:\n')
            for test in test_list:
                f.write('%s\n' % test)


def get_train_val_test_balance(fold_neg, fold_pos):

    # fold_path_neg = os.path.join(data_path, 'negative_cases3.txt')  # negative_cases.txt
    # fold_path_pos = os.path.join(data_path, 'positive_cases3.txt')
    fold_path_neg = fold_neg
    fold_path_pos = fold_pos
    f_pos = open(fold_path_pos, encoding='gbk')
    f_neg = open(fold_path_neg, encoding='gbk')
    txt_pos = []
    txt_neg = []
    for line in f_pos:
        txt_pos.append(line.strip())
    # train_index_pos = txt_pos.index('train:')
    # val_index_pos = txt_pos.index('val:')
    # test_index_pos = txt_pos.index('test:')

    # train_list_pos = txt_pos[train_index_pos+1: val_index_pos]
    # val_list_pos = txt_pos[val_index_pos+1: test_index_pos]
    # test_list_pos = txt_pos[test_index_pos+1:]

    for line in f_neg:
        txt_neg.append(line.strip())
    # train_index_neg = txt_neg.index('train:')
    # val_index_neg = txt_neg.index('val:')
    # test_index_neg = txt_neg.index('test:')
    #
    # train_list_neg = txt_neg[train_index_neg+1: val_index_neg]
    # val_list_neg = txt_neg[val_index_neg+1: test_index_neg]
    # test_list_neg = txt_neg[test_index_neg+1:]

    return txt_pos, txt_neg


def get_train_val_test(data_path, fold_index):
    fold_path = os.path.join(data_path, f'fold{fold_index}.txt')  # negative_cases.txt
    f = open(fold_path, encoding='gbk')
    txt = []
    for line in f:
        txt.append(line.strip())
    train_index = txt.index('train:')
    val_index = txt.index('val:')

    train_list = txt[train_index+1: val_index]
    val_list = txt[val_index+1:]

    return train_list, val_list


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent
