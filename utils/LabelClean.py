import abc
import os
from PIL import Image
import SimpleITK as sitk
import torch
import logging
import numpy as np
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt as distrans

JSRT_CLASS = ['lung', 'heart', 'clavicle']

class Cleaner():
    def __init__(self, model, trueloader,
                logger=logging.getLogger(__name__), **kwarg) -> None:
        self.model = model
        self.trueloader = trueloader
        self.logger = logger
        self.save_root = kwarg['cleaned']
        self.device = kwarg['device']
        self.num_class = kwarg['num_class']
        self.is_progress_bar = kwarg['is_progress_bar']
        self.startover = kwarg['startover']
        self.dataset = kwarg['dataset']
        self.show_clean_performance = kwarg['show_clean_performance']
        self.continue_clean = True
        if self.show_clean_performance:
            self.dsc_cleaned = np.zeros((self.num_class, len(self.trueloader)))
            self.dsc_noise = np.zeros((self.num_class, len(self.trueloader)))
        self.confidence_level = 0.5
    
    def __call__(self, train_loader, eval_loader, iter, sigma) -> None:
        kwargs = dict(desc="Iter {}".format(iter), leave=False,
                      disable=not self.is_progress_bar)
        self.logger.info('------ Cleaning Training Labels -------')
        diff = np.zeros(self.num_class)
        if iter == 1:
            confidence_level = self.confidence_level
        else:
            confidence_level = 0.5
        self.model.eval()

        # Estimate Bias
        length = 0
        for _, batch in enumerate(eval_loader):
            image, gt = batch['image'].to(self.device), batch['mask'].to(self.device)
            gt = gt.cpu().detach().squeeze().numpy()
            with torch.no_grad():
                output = self.model(image)
            pred = torch.sigmoid(output).cpu().detach().squeeze().numpy()
            for i in range(self.num_class):
                if self.dataset == 'Jsrt':
                    mask = (pred[i]>confidence_level).astype(float)
                    dis_pred = -distrans(mask) + distrans(1-mask)
                    dis_true = -distrans(gt[i]) + distrans(1-gt[i])
                else:
                    mask = (pred>confidence_level).astype(float)
                    dis_pred = -distrans(mask) + distrans(1-mask)
                    dis_true = -distrans(gt) + distrans(1-gt)
                if np.sum(dis_pred == -1) == 0 and np.sum(dis_true == -1) != 0:
                    diff[i] += 1#np.sum(dis_true == -1)
                    length += 1
                elif np.sum(dis_pred == -1) != 0 and np.sum(dis_true == -1) == 0:
                    diff[i] -= 1#np.sum(dis_pred == -1)
                    length += 1
                elif  np.sum(dis_pred == -1) == 0 and np.sum(dis_true == -1) == 0:
                    diff[i] += 0
                else:
                    diff[i] += np.mean(dis_pred[dis_true==1]-1)
                    length += 1
        diff = sigma * diff / length
        save_path = []
        for i in range(self.num_class):
            # if i != 0:
            #     diff = diff/sigma
            # if i == 2:
            #     diff = diff*0.7
            self.logger.info('Distance bias {:.2f}'.format(diff[i]))
            if abs(diff[i]) < 1:
                if self.dataset == 'Jsrt':
                    self.logger.info(JSRT_CLASS[i] + ' no need to be cleaned anymore.')
                    save_path.append(os.path.join(self.save_root, str(iter), JSRT_CLASS[i]))
                else:
                    self.logger.info('No need to be cleaned anymore.')
                    self.continue_clean = False
                    save_path.append(os.path.join(self.save_root, str(iter)))
                if not os.path.isdir(save_path[i]):
                    os.makedirs(save_path[i])
            else:
                if self.dataset == 'Jsrt':
                    save_path.append(os.path.join(self.save_root, str(iter), JSRT_CLASS[i]))
                else:
                    save_path.append(os.path.join(self.save_root, str(iter)))
                if not os.path.isdir(save_path[i]):
                    os.makedirs(save_path[i])
                self.logger.info('Cleaned labels will be saved in {}'.format(save_path[i]))
        
        # Clean Training Label
        with tqdm(len(train_loader.dataset), **kwargs) as pbar:
            for _, batch in enumerate(train_loader):
                image, target = batch['image'].to(self.device), batch['mask']
                index = batch['index']
                for b in range(len(index)):
                    id = index[b]
                    noise = target[b].squeeze().numpy()
                    with torch.no_grad():
                        output = self.model(image[b].unsqueeze(0))
                    pred = torch.sigmoid(output).cpu().detach().squeeze().numpy()
                    score = output.cpu().detach().squeeze().numpy()
                    for i in range(self.num_class): 
                        if abs(diff[i]) < 1:
                            if self.num_class > 1:
                                cleaned = pred[i]
                            else:
                                cleaned = pred
                        else:
                            if self.num_class > 1:
                                total = 1
                                for s in pred[i].shape:
                                    total *= s
                                if np.sum((pred[i]>confidence_level).astype(np.float32)) == 0 or np.sum((pred[i]>confidence_level).astype(np.float32)) == total:
                                    cleaned = pred[i] # cannot clean due to no boundary found
                                else:
                                    dis = -distrans(pred[i]>confidence_level) + distrans(1-(pred[i]>confidence_level))
                                    if diff[i] < 0:
                                        cleaned = self._shrink(score[i], dis, diff[i])
                                    else:
                                        cleaned = self._expand(score[i], dis, diff[i])
                            else:
                                total = 1
                                for s in pred.shape:
                                    total *= s
                                if np.sum((pred>confidence_level).astype(np.float32)) == 0 or np.sum((pred>confidence_level).astype(np.float32)) == total:
                                    cleaned = pred
                                else:
                                    dis = -distrans(pred>confidence_level) + distrans(1-(pred>confidence_level))
                                    if diff[i] < 0:
                                        cleaned = self._shrink(score, dis, diff[i])
                                    else:
                                        cleaned = self._expand(score, dis, diff[i])
                        if self.show_clean_performance:
                            if self.num_class > 1:
                                self.dsc_cleaned[i][id] = self._compare_with_clean(cleaned, i, id)
                                self.dsc_noise[i][id] = self._compare_with_clean(noise[i], i, id)
                            else:
                                self.dsc_cleaned[i][id] = self._compare_with_clean(cleaned, -1, id)
                                self.dsc_noise[i][id] = self._compare_with_clean(noise, -1, id)
                        file = train_loader.dataset.imgs[id]
                        new_path = os.path.join(save_path[i], file.split('/')[-1])
                        if self.dataset != 'Brats2020':
                            cleaned = Image.fromarray(((cleaned>0.5)*255).astype(np.uint8))
                            cleaned.save(new_path)
                        else:
                            cleaned = sitk.GetImageFromArray((cleaned>0.5).astype(np.int32))
                            sitk.WriteImage(cleaned, new_path)
                        if self.dataset == 'Jsrt':
                            if i == 0:
                                train_loader.dataset.gts_lung[id] = new_path
                            elif i == 1:
                                train_loader.dataset.gts_heart[id] = new_path
                            else:
                                train_loader.dataset.gts_clavicle[id] = new_path
                        else:
                            train_loader.dataset.gts[id] = new_path
                pbar.set_postfix()
                pbar.update(image.shape[0])
        if self.show_clean_performance:
            if self.dataset == 'Jsrt':
                noisy_classwise_mean = np.mean(self.dsc_noise, axis=1)*100
                noisy_classwise_std = np.std(self.dsc_noise, axis=1)*100
                noisy_total = np.mean(self.dsc_noise, axis=0)
                noisy_total_mean = np.mean(noisy_total)*100
                noisy_total_std = np.std(noisy_total)*100
                cleaned_classwise_mean = np.mean(self.dsc_cleaned, axis=1)*100
                cleaned_classwise_std = np.std(self.dsc_cleaned, axis=1)*100
                cleaned_total = np.mean(self.dsc_cleaned, axis=0)
                cleaned_total_mean = np.mean(cleaned_total)*100
                cleaned_total_std = np.std(cleaned_total)*100
                self.logger.info('Labels are improved by {:.2f}+-{:.2f}->{:.2f}+-{:.2f} (Lung) '
                                .format(noisy_classwise_mean[0], noisy_classwise_std[0], 
                                cleaned_classwise_mean[0], cleaned_classwise_std[0]) + 
                                '{:.2f}+-{:.2f}->{:.2f}+-{:.2f} (Heart) '
                                .format(noisy_classwise_mean[1], noisy_classwise_std[1], 
                                cleaned_classwise_mean[1], cleaned_classwise_std[1]) + 
                                '{:.2f}+-{:.2f}->{:.2f}+-{:.2f} (Clavicle) '
                                .format(noisy_classwise_mean[2], noisy_classwise_std[2], 
                                cleaned_classwise_mean[2], cleaned_classwise_std[2]) +
                                '{:.2f}+-{:.2f}->{:.2f}+-{:.2f} (Average) '
                                .format(noisy_total_mean, noisy_total_std, cleaned_total_mean, cleaned_total_std))
            else:
                noisy_total_mean = np.mean(self.dsc_noise)*100
                noisy_total_std = np.std(self.dsc_noise)*100
                cleaned_total_mean = np.mean(self.dsc_cleaned)*100
                cleaned_total_std = np.std(self.dsc_cleaned)*100
                self.logger.info('Labels are improved by {:.2f}+-{:.2f}->{:.2f}+-{:.2f}'
                                .format(noisy_total_mean, noisy_total_std, cleaned_total_mean, cleaned_total_std))
        if self.startover:
            self.model.apply(weight_init)

    def _shrink(self, score, distance_transform, distance):
        bias = np.max(abs(score[(distance_transform<0)&(distance_transform>=distance)]))
        score  = score - bias*np.exp(-distance_transform**2/(2*distance**2))
        return torch.sigmoid(torch.Tensor(score)).numpy()
    
    def _expand(self, score, distance_transform, distance):
        bias = np.max(abs(score[(distance_transform>0)&(distance_transform<=distance)]))
        score  = score + bias*np.exp(-distance_transform**2/(2*distance**2))
        return torch.sigmoid(torch.Tensor(score)).numpy()
    
    def _compare_with_clean(self, target, label, index):
        data = self.trueloader.dataset[index]
        gt = data['mask'].squeeze()
        if label >= 0:
            gt = gt[label].numpy()
        else:
            gt = gt.numpy()
        return dice(gt, target)


def weight_init(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def dice(gt, pred):
    pred = (pred > 0.5).astype(np.float32)
    inter = (gt*pred).sum().astype(np.float32)
    return (2*inter+1e-4)/(gt.sum()+pred.sum()+1e-4).astype(np.float32)

if __name__ == '__main__':
    pass

