from torchvision.utils import make_grid, save_image, draw_segmentation_masks
from net import UNet
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'
RANDOM_SEED = 123
FILE_NAME = 'cleaning.png'
PRED_NAME = 'pred.png'

class Visualizer():
    def __init__(self, length, logger, indices=None, dataset='Jsrt') -> None:
        self.dataset = dataset
        self.num = 6
        if not indices:
            np.random.seed(RANDOM_SEED)
            indices = np.random.randint(0, length, size=self.num)
        self.indices = indices
        self.data_grid = None
        self.pred_grid = None
        logger.info('')
    
    def append_loader(self, dataloader):
        images = []
        masks = []
        if self.dataset == 'Jsrt':
            for i in self.indices:
                data = dataloader.dataset[i]
                image = data['image'].squeeze().numpy()
                masks.append(data['mask'].squeeze().to(torch.bool))
                image = np.array([image]*3)
                image = torch.Tensor(image).to(torch.uint8)
                images.append(image)
            lung_with_mask = [draw_segmentation_masks(image, masks=mask, alpha=.6, colors=['blue', 'green', 'yellow']) 
                    for image, mask in zip(images, masks)]
        elif self.dataset == 'LIDC-IDRI':
            for i in self.indices:
                data = dataloader.dataset[i]
                image = data['image'].squeeze().numpy()
                masks.append(data['mask'].squeeze().to(torch.bool))
                image = np.array([image]*3)
                image = torch.Tensor(image).to(torch.uint8)
                images.append(image)
            lung_with_mask = [draw_segmentation_masks(image, masks=mask, alpha=.6, colors=['red']) 
                    for image, mask in zip(images, masks)]
        elif self.dataset == 'Brats2020':
            for i in self.indices:
                data = dataloader.dataset[i]
                image = data['image'].squeeze().numpy()
                mask = data['mask'].squeeze().to(torch.bool)
                slice = int(len(image)/2)
                image = image/np.max(image)
                image = image[slice]*255
                mask = mask[slice]
                masks.append(mask)
                image = np.array([image]*3)
                image = torch.Tensor(image).to(torch.uint8)
                images.append(image)
            lung_with_mask = [draw_segmentation_masks(image, masks=mask, alpha=.6, colors=['red']) 
                    for image, mask in zip(images, masks)]
        elif self.dataset == 'ISIC2017':
            for i in self.indices:
                data = dataloader.dataset[i]
                image = data['image'].squeeze().numpy()
                masks.append(data['mask'].squeeze().to(torch.bool))
                image = torch.Tensor(image).to(torch.uint8)
                images.append(image)
            lung_with_mask = [draw_segmentation_masks(image, masks=mask, alpha=.6, colors=['red']) 
                    for image, mask in zip(images, masks)]
        grid = make_grid(lung_with_mask)
        if self.data_grid == None:
            self.data_grid = grid
        else:
            self.data_grid = torch.hstack((self.data_grid, grid))

    def append_prediction(self, model, dataloader, device):
        preds = []
        images = []
        if model == None:
            for iter, batch in enumerate(dataloader):
                if iter > self.num-1:
                    break
                if self.dataset == 'Jsrt' or self.dataset == 'LIDC-IDRI':
                    image = batch['image'].squeeze().numpy()
                    pred = batch['mask'].squeeze()
                    preds.append(pred.to(torch.bool))
                    image = np.array([image]*3)
                    image = torch.Tensor(image).to(torch.uint8)
                    images.append(image)
                elif self.dataset == 'Brats2020':
                    image = batch['image'].squeeze().numpy()
                    pred = batch['mask'].squeeze()
                    slice = int(len(image)/2)
                    image = image/np.max(image)
                    image = image[slice]*255
                    pred = pred[slice]
                    preds.append(pred)
                    image = np.array([image]*3)
                    image = torch.Tensor(image).to(torch.uint8)
                    images.append(image)
                elif self.dataset == 'ISIC2017':
                    image = batch['image'].squeeze().numpy()
                    pred = batch['mask'].squeeze()
                    preds.append(pred.to(torch.bool))
                    image = torch.Tensor(image).to(torch.uint8)
                    images.append(image)
        else:
            model.eval()
            for iter, batch in enumerate(dataloader):
                if iter > self.num-1:
                    break
                image = batch['image'].to(device)
                with torch.no_grad():
                    output = model(image)
                pred = torch.sigmoid(output)
                pred = (pred > 0.5).float().detach().cpu()
                if self.dataset == 'Jsrt' or self.dataset == 'LIDC-IDRI':
                    image = batch['image'].squeeze().numpy()
                    preds.append(pred.squeeze().to(torch.bool))
                    image = np.array([image]*3)
                    image = torch.Tensor(image).to(torch.uint8)
                    images.append(image)
                elif self.dataset == 'Brats2020':
                    image = batch['image'].squeeze().numpy()
                    pred = pred.squeeze().to(torch.bool)
                    slice = int(len(image)/2)
                    image = image/np.max(image)
                    image = image[slice]*255
                    pred = pred[slice]
                    preds.append(pred)
                    image = np.array([image]*3)
                    image = torch.Tensor(image).to(torch.uint8)
                    images.append(image)
                elif self.dataset == 'ISIC2017':
                    image = batch['image'].squeeze().numpy()
                    preds.append(pred.squeeze().to(torch.bool))
                    image = torch.Tensor(image).to(torch.uint8)
                    images.append(image)
                
        if self.dataset == 'Jsrt':
            lung_with_mask = [draw_segmentation_masks(image, masks=mask, alpha=.6, colors=['blue', 'green', 'yellow']) 
                        for image, mask in zip(images, preds)]
        else:
            lung_with_mask = [draw_segmentation_masks(image, masks=mask, alpha=.5, colors=['red']) 
                    for image, mask in zip(images, preds)]
        grid = make_grid(lung_with_mask)
        if self.pred_grid == None:
            self.pred_grid = grid
        else:
            self.pred_grid = torch.hstack((self.pred_grid, grid))
    
    def draw(self, mode='data'):
        if mode == 'data':
            self.__show(self.data_grid)
        else:
            self.__show(self.pred_grid)

    def save(self, path):
        if len(self.data_grid) > 0:
            result = np.transpose(self.data_grid.numpy(), (1,2,0))
            result = Image.fromarray(result.astype(np.uint8))
            result.save(os.path.join(path, FILE_NAME))

        if len(self.pred_grid) > 0:
            result = np.transpose(self.pred_grid.numpy(), (1,2,0))
            result = Image.fromarray(result.astype(np.uint8))
            result.save(os.path.join(path, PRED_NAME))

    def __show(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])




