import os
import random
from glob import glob
import codecs
from skimage import transform
import pydicom
import numpy as np
import xlrd
from tqdm import tqdm
import time


def split_train_val_test(data_path, fold):

    id_list = os.listdir(data_path)
    random.shuffle(id_list)
    id_list = [i.split('.')[0] for i in id_list]
    train_list = id_list[:400]
    val_list = id_list[400:500]
    test_list = id_list[500:]
    with codecs.open('fold_{}.txt'.format(fold), encoding='utf-8', mode='w') as f:
        f.write('train:\n')
        for train in train_list:
            f.write('%s\n' % train)

        f.write('val:\n')
        for val in val_list:
            f.write('%s\n' % val)

        f.write('test:\n')
        for test in test_list:
            f.write('%s\n' % test)


def split_train_val_test_balance(data_path, xlsx_path, fold, error_list):
    workbook = xlrd.open_workbook(xlsx_path)
    worksheet = workbook.sheet_by_index(0)
    xlsx_id_list = worksheet.col_values(0)[1:]
    xlsx_id_list = [int(i) for i in xlsx_id_list]
    xlsx_label_list = worksheet.col_values(4)[1:]
    xlsx_cspca = worksheet.col_values(6)[1:]
    id_list = os.listdir(data_path)
    for error_id in error_list:
        if '{}.npz'.format(error_id) in id_list:
            id_list.remove('{}.npz'.format(error_id))
            print(error_id)
    print(len(id_list))
    random.shuffle(id_list)
    pos_list = []
    neg_list = []
    e_list = []
    for case_id in id_list:
        id_int = int(case_id.split('.')[0])
        id_ = case_id.split('.')[0]
        case_index = xlsx_id_list.index(id_int)
        label = xlsx_label_list[case_index]
        cspca = xlsx_cspca[case_index]
        if label == 1 and cspca == '':
            e_list.append(case_id)
            continue
        label_cspca = cspca if label != 0 else 0
        if int(label_cspca) == 1:
            pos_list.append(id_)
        else:
            neg_list.append(id_)
    print(e_list)

    pos_train_list = pos_list[: 270]
    pos_val_list = pos_list[270: 340]
    pos_test_list = pos_list[340: ]
    with codecs.open('fold_{}_positive.txt'.format(fold), encoding='utf-8', mode='w') as f:
        f.write('train:\n')
        for train in pos_train_list:
            f.write('%s\n' % train)

        f.write('val:\n')
        for val in pos_val_list:
            f.write('%s\n' % val)

        f.write('test:\n')
        for test in pos_test_list:
            f.write('%s\n' % test)

    neg_train_list = neg_list[: 130]
    neg_val_list = neg_list[130: 160]
    neg_test_list = neg_list[160:]
    with codecs.open('fold_{}_negative.txt'.format(fold), encoding='utf-8', mode='w') as f:
        f.write('train:\n')
        for train in neg_train_list:
            f.write('%s\n' % train)

        f.write('val:\n')
        for val in neg_val_list:
            f.write('%s\n' % val)

        f.write('test:\n')
        for test in neg_test_list:
            f.write('%s\n' % test)


def split_train_val_balance(data_path, xlsx_path, fold, error_list):

    workbook = xlrd.open_workbook(xlsx_path)
    worksheet = workbook.sheet_by_index(0)
    xlsx_id_list = worksheet.col_values(0)[1:]
    xlsx_id_list = [int(i) for i in xlsx_id_list]
    xlsx_label_list = worksheet.col_values(4)[1:]
    xlsx_cspca = worksheet.col_values(6)[1:]
    id_list = os.listdir(data_path)
    for error_id in error_list:
        if '{}.npz'.format(error_id) in id_list:
            id_list.remove('{}.npz'.format(error_id))
            print(error_id)
    print(len(id_list))
    random.shuffle(id_list)
    pos_list = []
    neg_list = []
    e_list = []
    for case_id in id_list:
        id_int = int(case_id.split('.')[0])
        id_ = case_id.split('.')[0]
        case_index = xlsx_id_list.index(id_int)
        label = xlsx_label_list[case_index]
        cspca = xlsx_cspca[case_index]
        if label == 1 and cspca == '':
            e_list.append(case_id)
            continue
        label_cspca = cspca if label != 0 else 0
        if int(label_cspca) == 1:
            pos_list.append(id_)
        else:
            neg_list.append(id_)
    print(e_list)
    l = len(pos_list)
    pos_train_list = pos_list[:int(l/6*5)]
    pos_val_list = pos_list[int(l/6*5):]
    with codecs.open('fold_{}_positive.txt'.format(fold), encoding='utf-8', mode='w') as f:
        f.write('train:\n')
        for train in pos_train_list:
            f.write('%s\n' % train)

        f.write('val:\n')
        for val in pos_val_list:
            f.write('%s\n' % val)

    l = len(neg_list)
    neg_train_list = neg_list[:int(l/3*2)]
    neg_val_list = neg_list[int(l/3*2):]
    with codecs.open('fold_{}_negative.txt'.format(fold), encoding='utf-8', mode='w') as f:
        f.write('train:\n')
        for train in neg_train_list:
            f.write('%s\n' % train)

        f.write('val:\n')
        for val in neg_val_list:
            f.write('%s\n' % val)


def get_train_val_test(data_path, fold):

    fold_path = os.path.join(data_path, 'fold_{}.txt'.format(fold))
    f = open(fold_path, encoding='gbk')
    txt = []
    for line in f:
        txt.append(line.strip())
    train_index = txt.index('train:')
    val_index = txt.index('val:')
    test_index = txt.index('test:')

    train_list = txt[train_index+1: val_index]
    val_list = txt[val_index+1: test_index]
    test_list = txt[test_index+1:]

    return train_list, val_list, test_list


def get_train_val_test_balance(data_path, fold):

    fold_path_neg = os.path.join(data_path, 'fold_{}_negative.txt'.format(fold))
    fold_path_pos = os.path.join(data_path, 'fold_{}_positive.txt'.format(fold))
    f_pos = open(fold_path_pos, encoding='gbk')
    f_neg = open(fold_path_neg, encoding='gbk')
    txt_pos = []
    txt_neg = []
    for line in f_pos:
        txt_pos.append(line.strip())
    train_index_pos = txt_pos.index('train:')
    val_index_pos = txt_pos.index('val:')
    test_index_pos = txt_pos.index('test:')

    train_list_pos = txt_pos[train_index_pos+1: val_index_pos]
    val_list_pos = txt_pos[val_index_pos+1: test_index_pos]
    test_list_pos = txt_pos[test_index_pos+1:]

    for line in f_neg:
        txt_neg.append(line.strip())
    train_index_neg = txt_neg.index('train:')
    val_index_neg = txt_neg.index('val:')
    test_index_neg = txt_neg.index('test:')

    train_list_neg = txt_neg[train_index_neg+1: val_index_neg]
    val_list_neg = txt_neg[val_index_neg+1: test_index_neg]
    test_list_neg = txt_neg[test_index_neg+1:]

    return [train_list_pos, val_list_pos, test_list_pos], [train_list_neg, val_list_neg, test_list_neg]


def get_train_val_balance(data_path, fold):

    fold_path_neg = os.path.join(data_path, 'fold_{}_negative.txt'.format(fold))
    fold_path_pos = os.path.join(data_path, 'fold_{}_positive.txt'.format(fold))
    f_pos = open(fold_path_pos, encoding='gbk')
    f_neg = open(fold_path_neg, encoding='gbk')
    txt_pos = []
    txt_neg = []
    for line in f_pos:
        txt_pos.append(line.strip())
    train_index_pos = txt_pos.index('train:')
    val_index_pos = txt_pos.index('val:')

    train_list_pos = txt_pos[train_index_pos+1: val_index_pos]
    val_list_pos = txt_pos[val_index_pos+1:]

    for line in f_neg:
        txt_neg.append(line.strip())
    train_index_neg = txt_neg.index('train:')
    val_index_neg = txt_neg.index('val:')

    train_list_neg = txt_neg[train_index_neg+1: val_index_neg]
    val_list_neg = txt_neg[val_index_neg+1: ]

    return [train_list_pos, val_list_pos], [train_list_neg, val_list_neg]


def get_train_val_balance_crossval(data_path, fold, n_splits, k_idx):

    fold_path_neg = os.path.join(data_path, 'fold_{}_negative_k{}.txt'.format(fold, n_splits))
    fold_path_pos = os.path.join(data_path, 'fold_{}_positive_k{}.txt'.format(fold, n_splits))
    f_pos = open(fold_path_pos, encoding='gbk')
    f_neg = open(fold_path_neg, encoding='gbk')
    txt_pos = []
    txt_neg = []
    for line in f_pos:
        txt_pos.append(line.strip())
    train_index_pos = [index for index, value in enumerate(txt_pos) if value == 'train:'][k_idx]
    val_index_pos = [index for index, value in enumerate(txt_pos) if value == 'val:'][k_idx]
    test_index_pos = [index for index, value in enumerate(txt_pos) if value == 'test:'][k_idx]

    train_list_pos = txt_pos[train_index_pos+1: val_index_pos]
    val_list_pos = txt_pos[val_index_pos+1: test_index_pos]

    for line in f_neg:
        txt_neg.append(line.strip())
    train_index_neg = [index for index, value in enumerate(txt_neg) if value == 'train:'][k_idx]
    val_index_neg = [index for index, value in enumerate(txt_neg) if value == 'val:'][k_idx]
    test_index_neg = [index for index, value in enumerate(txt_neg) if value == 'test:'][k_idx]

    train_list_neg = txt_neg[train_index_neg+1: val_index_neg]
    val_list_neg = txt_neg[val_index_neg+1: test_index_neg]

    return [train_list_pos, val_list_pos], [train_list_neg, val_list_neg]


def save2npz(data_path, save_path, new_size, id_list_xls, label_list_xls, ga_list_xls, cspca_list_xls):
    image = pydicom.dcmread(data_path)
    case_id = os.path.basename(data_path).split('.')[0]
    img_array = image.pixel_array[..., 0]
    new_size.insert(0, img_array.shape[0])
    img_zoom = transform.resize(img_array, new_size, order=3)
    label_index = id_list_xls.index(int(case_id))
    np.savez_compressed('{}/{}.npz'.format(save_path, case_id), volume=img_zoom, label=int(label_list_xls[label_index]),
                        Gleason=ga_list_xls[label_index], CsPCa=cspca_list_xls[label_index], origin_size=img_array.shape)
    print("id:{} --- label:{} --- gleason:{} --- cspca:{}"
          .format(case_id, int(label_list_xls[label_index]), ga_list_xls[label_index], cspca_list_xls[label_index]))


def find_duplicates(nums):
    unique_set = set()
    duplicates = []
    for num in nums:
        if num in unique_set:
            duplicates.append(num)
        else:
            unique_set.add(num)
    return duplicates


if __name__ == '__main__':
    data_path = '/hy-tmp/datasets/144%144%200ROINPZ'
    xlsx_path = '/hy-tmp/code/ProstateSeg_yun/video/code/utils/prostate.xlsx'
    save_path = '/hy-tmp/code/ProstateSeg_yun/video/code/utils'
    error_list = ['004', '010', '088', '092', '153', '202', '240', '283', '373', '419', '506', '515', '521', '538',
                  '601', '639']
    split_train_val_test_balance(data_path, xlsx_path, 'BMode612', error_list)

