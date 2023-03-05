from tqdm import tqdm
import torch
import logging
from .dice_loss import DiceCoeff
import numpy as np

class Evaluator():
    def __init__(self, model, writer, metric='dice',
                 device=torch.device('cpu'), logger=logging.getLogger(__name__),
                 save_file = False, **kwargs) -> None:
        self.model = model
        self.writer = writer
        self.device = device
        self.logger = logger
        self.num_class = kwargs['num_class']
        self.dataset = kwargs['dataset']
        self.save_file = save_file
    
    def __call__(self, eval_loader, epoch) -> None:
        self.model.eval()
        dcnt = len(eval_loader)
        dval = np.zeros((self.num_class, dcnt))
        for iter, batch in enumerate(eval_loader):
            image, target = batch['image'].to(self.device), batch['mask'].to(self.device)
            index = batch['index']
            with torch.no_grad():
                output = self.model(image)
            pred = torch.sigmoid(output)
            #print(torch.unique(pred))
            pred = (pred > 0.5).float()
            if self.num_class == 1:
                dval[0][iter] = self._dice_score(pred, target, -1, self.device).item()
                #dval[0][index[0]] = self.dice(target.cpu().detach().squeeze().numpy(), pred.cpu().detach().squeeze().numpy())
            else:
                for j in range(self.num_class):
                    dval[j][index[0]] = self._dice_score(pred, target, j, self.device).item()
        if self.dataset == 'Jsrt':
            classwise_mean = np.mean(dval, axis=1)*100
            classwise_std = np.std(dval, axis=1)*100
            total = np.mean(dval, axis=0)
            total_mean = np.mean(total)*100
            total_std = np.std(total)*100
            if not self.save_file:
                self.logger.info('Epoch {:d} Lung--Heart--Clavicle--Mean ({:.2f}--{:.2f}--{:.2f}--{:.2f})'
                                .format(epoch, classwise_mean[0], 
                                        classwise_mean[1], classwise_mean[2], total_mean))
                self.writer.add_scalar('eval_acc/Lung', classwise_mean[0], epoch)
                self.writer.add_scalar('eval_acc/Heart', classwise_mean[1], epoch)
                self.writer.add_scalar('eval_acc/Clavicle', classwise_mean[2], epoch)
                self.writer.add_scalar('eval_acc/Average', total_mean, epoch)
            else:
                self.logger.info('Epoch {:d}: Lung {:.2f}+-{:.2f} '.format(epoch, classwise_mean[0], classwise_std[0]) +
                              'Heart {:.2f}+-{:.2f} '.format(classwise_mean[1], classwise_std[1]) +
                              'Clavicle {:.2f}+-{:.2f} '.format(classwise_mean[2], classwise_std[2]) +
                              'Average {:.2f}+-{:.2f}'.format(total_mean, total_std))
            return classwise_mean
        else:
            mean = np.mean(dval[0])*100
            std = np.std(dval[0])*100
            if not self.save_file:
                self.logger.info('Epoch {:d} Dice {:.2f}'.format(epoch, mean))
                self.writer.add_scalar('eval_acc', mean, epoch)
            else:
                self.logger.info('Epoch {:d}: {:.2f}+-{:.2f}'.format(epoch, mean, std))
            return mean
    
    def cross_eval(self, dsc_list, dataset):
        dsc = np.array(dsc_list) # model_num * [class_num]
        if dataset == 'Jsrt':
            classwise_mean = np.mean(dsc, axis = 0)
            classwise_std = np.std(dsc, axis = 0)
            total = np.mean(dsc, axis = 1)
            total_mean = np.mean(total)
            total_std = np.std(total)
            self.logger.info('Cross Eval: [Lung] {:.2f}+-{:.2f} '.format(classwise_mean[0], classwise_std[0]) +
                              '[Heart] {:.2f}+-{:.2f} '.format(classwise_mean[1], classwise_std[1]) +
                              '[Clavicle] {:.2f}+-{:.2f} '.format(classwise_mean[2], classwise_std[2]) +
                              '[Average] {:.2f}+-{:.2f}'.format(total_mean, total_std))
        else:
            mean = np.mean(dsc)
            std = np.std(dsc)
            self.logger.info('Cross Eval: {:.2f}+-{:.2f}'.format(mean, std))
    
    def dice(self, gt, pred):
        eps = 1e-5
        #pred = (pred > 0.5).astype(np.float)
        inter = (gt*pred).sum().astype(np.float)
        return (2*inter+eps)/(gt.sum()+pred.sum() + eps).astype(np.float)

    def _dice_score(self, input, target, l, device):
        """Dice coeff for batches"""
        if input.is_cuda:
            s = torch.FloatTensor(1).to(device).zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            if l >= 0:
                s = s + DiceCoeff().forward(c[0][l], c[1][l])
            else:
                s = s + DiceCoeff().forward(c[0], c[1])

        return s / (i + 1)