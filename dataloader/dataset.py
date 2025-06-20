import torch
import os
import numpy as np
from torch.utils.data import Dataset
# import cv2
# import SimpleITK as sitk
import pandas as pd

class PD3C_B_E(Dataset):
    def __init__(self, case_list, b_root_path, e_root_path, transform=None):
        self.case_list = case_list
        self.b_root_path = b_root_path
        self.e_root_path = e_root_path
        # self.box_root_path = box_root_path
        # self.box_list = box_list if box_list else []
        self.transform = transform

        excel_path = '/prostate.xlsx'
        df = pd.read_excel(excel_path, usecols=['ID', 'class', 'csPCa'])
        self.excel_inform = df.to_numpy()
        self.id_list = self.excel_inform[:, 0]

    def __getitem__(self, index):

        id_ = self.case_list[index]
        bmode_path = os.path.join(self.b_root_path, self.case_list[index] + '_transverse.npz')
        swe_path = os.path.join(self.e_root_path, self.case_list[index] + '_sagittal.npz')
        data = np.load(bmode_path, allow_pickle=True)
        bmode = data['image']  # [:, :, :, 0:1]
        data = np.load(swe_path, allow_pickle=True)
        swe = data['image']  # .transpose(2, 1, 0)

        id2col = np.where(self.id_list == int(id_))[0]
        single_img_inform = self.excel_inform[id2col][0]
        if single_img_inform[1] == 1 and single_img_inform[2] == 1:
            cspca = 1
        else:
            cspca = 0
        sample = {'name': id_, 'volume1': bmode, 'volume2': swe,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PD3C(Dataset):
    def __init__(self, case_list, root_path, transform=None):
        self.case_list = case_list
        self.root_path = root_path
        # self.box_root_path = box_root_path
        # self.box_list = box_list if box_list else []
        self.transform = transform

        excel_path = '/prostate.xlsx'
        df = pd.read_excel(excel_path, usecols=['ID', 'class', 'csPCa'])
        self.excel_inform = df.to_numpy()
        self.id_list = self.excel_inform[:, 0]

    def __getitem__(self, index):

        id_ = self.case_list[index]
        if 'transverse' in self.root_path:
            img_path = os.path.join(self.root_path, self.case_list[index] + '_transverse.npz')
        else:
            img_path = os.path.join(self.root_path, self.case_list[index] + '_sagittal.npz')
        data = np.load(img_path, allow_pickle=True)
        data = data['image']  # [:, :, :, 0:1]

        id2col = np.where(self.id_list == int(id_))[0]
        single_img_inform = self.excel_inform[id2col][0]
        if single_img_inform[1] == 1 and single_img_inform[2] == 1:
            cspca = 1
        else:
            cspca = 0

        sample = {'name': id_, 'volume': data,
                      'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PD3C2(Dataset):
    def __init__(self, case_list, root_path, transform=None):
        self.case_list = case_list
        self.root_path = root_path
        # self.box_root_path = box_root_path
        # self.box_list = box_list if box_list else []
        self.transform = transform

        excel_path = '/prostate.xlsx'
        df = pd.read_excel(excel_path, usecols=['ID', 'class', 'csPCa'])
        self.excel_inform = df.to_numpy()
        self.id_list = self.excel_inform[:, 0]

    def __getitem__(self, index):

        id_ = self.case_list[index]
        if 'transverse' in self.root_path:
            img_path = os.path.join(self.root_path, self.case_list[index] + '_transverse.npz')
        else:
            img_path = os.path.join(self.root_path, self.case_list[index] + '_sagittal.npz')
        data = np.load(img_path, allow_pickle=True)
        data = data['image']  # [:, :, :, 0:1]

        id2col = np.where(self.id_list == int(id_))[0]
        single_img_inform = self.excel_inform[id2col][0]
        if single_img_inform[1] == 1 and single_img_inform[2] == 1:
            cspca = 1
        else:
            cspca = 0

        sample = {'volume': data,
                'labels': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample['volume'], sample['labels']

    def __len__(self):
        return len(self.case_list)


class PD1C_B_E_gary_bbox(Dataset):
    def __init__(self, case_list, b_root_path, e_root_path, box_root_path, transform=None):
        self.case_list = case_list
        self.b_root_path = b_root_path
        self.e_root_path = e_root_path
        self.box_root_path = box_root_path
        self.transform = transform

    def __getitem__(self, index):
        data_path = os.path.join(self.b_root_path, self.case_list[index] + '.npz')
        swe_path = os.path.join(self.e_root_path, self.case_list[index] + '.npy')
        box_path = os.path.join(self.box_root_path, self.case_list[index] + '.npy')
        case_id = self.case_list[index]
        data = np.load(data_path, allow_pickle=True)
        volume = data['volume1']
        benign_malignant = data['label']
        # c z y x -> c x y z
        volume1 = volume.transpose(2, 1, 0)
        # volume = volume[0:1, :, :, :]
        swe = np.load(swe_path).transpose(2, 1, 0)
        box = np.load(box_path).transpose(2, 1, 0).astype(np.int8)
        # swe = np.expand_dims(swe, axis=0)
        name = self.case_list[index]
        mask = data['mask'].transpose(2, 1, 0)
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': name, 'volume1': volume1, 'volume2': swe, 'box': box,
                  'cspca': 1}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class NormaKScore(object):
    def __init__(self, volume_key='volume'):
        self.volume_key = volume_key

    def __call__(self, sample):
        image_array = sample[self.volume_key]
        arr = image_array.reshape(-1)
        arr_mean = np.mean(arr)
        arr_var = np.var(arr)
        image_array = (image_array - arr_mean) / (arr_var + 1e-6)
        sample[self.volume_key] = image_array
        return sample


class ToTensor(object):
    def __init__(self, box_prefix=False):
        self.box_prefix = box_prefix

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']

        if self.box_prefix:
            box = np.expand_dims(sample['box'], axis=self.channels)

        sample['volume1'] = torch.from_numpy(volume1.copy()).unsqueeze(0)
        sample['volume2'] = torch.from_numpy(volume2.copy()).unsqueeze(0)  # .permute(3, 0, 1, 2)
        if self.box_prefix:
            sample['box'] = torch.from_numpy(box.copy())
        return sample


class ToTensor1(object):
    def __init__(self):
        self.prefix = 'volume'

    def __call__(self, sample):
        volume = sample[self.prefix]

        sample[self.prefix] = torch.from_numpy(volume).unsqueeze(0).float() # .permute(3, 0, 1, 2)
        return sample


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def next(self):
        sample = self.sample
        self.preload()
        return sample

    def preload(self):
        try:
            self.sample = next(self.loader)
        except StopIteration:
            self.sample = None
            return



