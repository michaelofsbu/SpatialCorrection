from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import monai.transforms as transforms
from monai.transforms.transform import MapTransform
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

    # def __getitem__(self, index):
    #     assert self.flair[index].split('/')[-1][:3] == self.gts[index].split('/')[-1][:3], 'file not matching'
    #     flair = sitk.GetArrayFromImage(sitk.ReadImage(self.flair[index])).astype(np.float32)
    #     # t1 = sitk.GetArrayFromImage(sitk.ReadImage(self.t1[index])).astype(np.float32)
    #     # t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2[index])).astype(np.float32)
    #     # t1ce = sitk.GetArrayFromImage(sitk.ReadImage(self.t1ce[index])).astype(np.float32)
    #     gt = sitk.GetArrayFromImage(sitk.ReadImage(self.gts[index])).astype(np.uint8)
    #     gt[gt>0] = 1
    #     if self.mode == 'train':
    #         transform = self._get_brats2021_train_transform(32, 1.0, 1.0)
    #     else:
    #         transform = self._get_brats2021_infer_transform()
    #     #item = transform({'flair':flair, 't1':t1, 't1ce':t1ce, 't2':t2, 'label':gt})
    #     item = transform({'flair':flair, 'label':gt})
    #     if self.mode == 'train':
    #         item = item[0]
        
    #     return {'image': item['image'], 'mask': item['label'], 'index': index}

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

    def _get_brats2021_base_transform(self):
        # base_transform = [
        #     # [B, H, W, D] --> [B, C, H, W, D]
        #     transforms.AddChanneld(keys=['flair', 't1', 't1ce', 't2', 'label']),      
        #     transforms.Orientationd(keys=['flair', 't1', 't1ce', 't2', 'label'], axcodes="RAS"),  
        #     RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),
        #     transforms.ConcatItemsd(keys=['flair', 't1', 't1ce', 't2'], name='image', dim=0),
        #     transforms.DeleteItemsd(keys=['flair', 't1', 't1ce', 't2']),
        #     #transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
        # ]
        base_transform = [
            # [B, H, W, D] --> [B, C, H, W, D]
            transforms.AddChanneld(keys=['flair', 'label']),      
            transforms.Orientationd(keys=['flair', 'label'], axcodes="RAS"),  
            RobustZScoreNormalization(keys=['flair']),
            transforms.ConcatItemsd(keys=['flair'], name='image', dim=0),
            transforms.DeleteItemsd(keys=['flair']),
            #transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
        ]
        return base_transform


    def _get_brats2021_train_transform(self, patch_size=128, pos_ratio=1.0, neg_ratio=1.0):
        base_transform = self._get_brats2021_base_transform()
        data_aug = [
            # crop
            transforms.RandCropByPosNegLabeld(
                keys=["image", 'label'], 
                label_key='label',
                spatial_size=[patch_size] * 3, 
                pos=pos_ratio, 
                neg=neg_ratio, 
                num_samples=1),

            # spatial aug
            transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

            # intensity aug
            transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
            transforms.RandGaussianSmoothd(
                keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

            # other stuff
            transforms.EnsureTyped(keys=["image", 'label']),
        ]
        return transforms.Compose(base_transform + data_aug)
        #return transforms.Compose(data_aug)


    def _get_brats2021_infer_transform(self):
        base_transform = self._get_brats2021_base_transform()
        infer_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
        return transforms.Compose(base_transform + infer_transform)
        #return transforms.Compose(infer_transform)


class RobustZScoreNormalization(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key] > 0

            lower = np.percentile(d[key][mask], 0.2)
            upper = np.percentile(d[key][mask], 99.8)

            d[key][mask & (d[key] < lower)] = lower
            d[key][mask & (d[key] > upper)] = upper

            y = d[key][mask]
            d[key] -= y.mean()
            d[key] /= y.std()

        return d