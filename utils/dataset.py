from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from torch.nn import Upsample
import logging
import cv2
from PIL import Image
import SimpleITK as sitk
from random import sample
from scipy.ndimage.morphology import distance_transform_edt as distrans


class ISICDataset(Dataset):
    def __init__(self, datapath, gtpath, size=None, mode='train'):
        self.imgs = sorted([datapath + f for f in listdir(datapath) if not f.startswith('.')])
        self.gts = sorted([gtpath + f for f in listdir(gtpath) if not f.startswith('.')])
        self.size = size
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        #assert self.imgs[index].split('/')[-1]==self.gts[index].split('/')[-1], "Filename {} and {} incompatible".format(self.imgs[index], self.gts[index])
        img = Image.open(self.imgs[index])
        label = Image.open(self.gts[index])
        if self.size:
            img = img.resize(self.size, resample=Image.NEAREST)
            label = label.resize(self.size, resample=Image.NEAREST)
        img = np.asarray(img)
        label = np.asarray(label)/255
        img = torch.Tensor(img.copy())
        if len(img.shape) == 3:
            img = img.permute(2, 0, 1)
        else:
            img = img.unsqueeze(0)
        label = torch.Tensor(label)
        label = label.unsqueeze(0)
        return {'image': img, 'mask': label, 'index': index}

class WeightDataset(Dataset):
    def __init__(self, datapath, labelpath, size=None, mode=None, scale=6):
        self.imgs = sorted([datapath + f for f in listdir(datapath) if not f.startswith('.')])
        self.gts = sorted([labelpath + f for f in listdir(labelpath) if not f.startswith('.')])
        self.size = size
        self.mode = mode
        self.scale = scale
        logging.info(f'Creating dataset with {len(self.imgs)} examples')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        label = Image.open(self.gts[index])
        if self.size:
            img = img.resize(self.size, resample=Image.NEAREST)
            label = label.resize(self.size, resample=Image.NEAREST)
        img = np.asarray(img)
        label = np.asarray(label)/255
        if self.mode == 'train':
            dis = distrans(label) + distrans(1-label)
            weight = np.where(dis<self.scale, 0, 1)
            img = torch.Tensor(img.copy())
            #img = img.permute(2, 0, 1)
            label = torch.Tensor(label)
            weight = torch.Tensor(weight)
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)
            weight = weight.unsqueeze(0)
            return {'image': img, 'mask': label, 'weight': weight}
        else:
            img = torch.Tensor(img.copy())
            #img = img.permute(2, 0, 1)
            label = torch.Tensor(label)
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)
            return {'image': img, 'mask': label}

class JSRTDataset(Dataset):
    def __init__(self, datapath, lungpath, heartpath, claviclepath, mode, size=None):
        self.mode = mode
        self.imgs = sorted([datapath + f for f in listdir(datapath)])
        self.gts_lung = sorted([lungpath + f for f in listdir(lungpath)])
        self.gts_heart = sorted([heartpath + f for f in listdir(heartpath)])
        self.gts_clavicle = sorted([claviclepath + f for f in listdir(claviclepath)])
        if size:
            index = sample(list(range(len(self.imgs))), size)
            self.imgs = [self.imgs[i] for i in index]
            self.gts_lung = [self.gts_lung[i] for i in index]
            self.gts_heart = [self.gts_heart[i] for i in index]
            self.gts_clavicle = [self.gts_clavicle[i] for i in index]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.imgs[index]))
        lung = np.asarray(Image.open(self.gts_lung[index]))/255
        heart = np.asarray(Image.open(self.gts_heart[index]))/255
        clavicle = np.asarray(Image.open(self.gts_clavicle[index]))/255
        label = np.array([lung, heart, clavicle])
        img = torch.Tensor(img.copy())
        label = torch.Tensor(label)
        img = img.unsqueeze(0)
        return {'image': img, 'mask': label, 'index': index}

class LIDCDataset(Dataset):
    def __init__(self, datapath, gtpath, mode, size=None):
        self.mode = mode
        self.imgs = sorted([datapath + f for f in listdir(datapath)])
        self.gts = sorted([gtpath + f for f in listdir(gtpath)])
        if size:
            index = sample(list(range(len(self.imgs))), size)
            self.imgs = [self.imgs[i] for i in index]
            self.gts = [self.gts[i] for i in index]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.imgs[index]))
        gt = np.asarray(Image.open(self.gts[index]))/255
        img = torch.Tensor(img.copy())
        label = torch.Tensor(gt).unsqueeze(0)
        img = img.unsqueeze(0)
        return {'image': img, 'mask': label, 'index': index}


class BratsDataset(Dataset):
    def __init__(self, datapath, gtpath, mode):
        self.mode = mode
        self.imgs = sorted([datapath + f for f in listdir(datapath) if f.endswith('_flair.nii.gz')])
        # self.t1 = sorted([datapath + f for f in listdir(datapath) if f.endswith('_t1.nii.gz')])
        # self.t2 = sorted([datapath + f for f in listdir(datapath) if f.endswith('_t2.nii.gz')])
        # self.t1ce = sorted([datapath + f for f in listdir(datapath) if f.endswith('_t1ce.nii.gz')])
        self.gts = sorted([gtpath + f for f in listdir(gtpath)])
        self.resize = Upsample(size=(64,128,128))
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        assert self.imgs[index].split('/')[-1][:3] == self.gts[index].split('/')[-1][:3], 'file not matching'
        flair = sitk.GetArrayFromImage(sitk.ReadImage(self.imgs[index])).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(self.gts[index])).astype(np.uint8)
        gt[gt>0] = 1
        image = torch.Tensor(flair).unsqueeze(0).unsqueeze(0)
        label = torch.Tensor(gt).unsqueeze(0).unsqueeze(0)
        image = self.resize(image).squeeze(0)
        label = self.resize(label).squeeze(0)
        return {'image': image, 'mask': label, 'index': index}