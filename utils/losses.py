import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

def get_loss_f(**kwarg):
    dataset = kwarg['dataset']
    if dataset == 'Jsrt':
        return WeightedBCE(kwarg['device'], kwarg['num_class'])
    if dataset == 'Brats2020':
        return SoftDiceBCEWithLogitsLoss()
        #return nn.BCEWithLogitsLoss()
    if dataset == 'ISIC2017':
        return nn.BCEWithLogitsLoss()
    if dataset == 'LIDC-IDRI':
        #return nn.BCEWithLogitsLoss()
        return SCELoss(kwarg['device'], dataset)

class SCELoss(torch.nn.Module):
    def __init__(self, device, dataset, alpha=0.5, beta=1):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if dataset == 'Jsrt':       
            self.bce = WeightedBCE(device, 3)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # CCE
        ce = self.bce(outputs, targets)

        # RCE
        labels = torch.clamp(targets, min=1e-5, max=1.0-1e-5)
        pred = torch.sigmoid(outputs)
        #rce = self.bce(targets, pred)
        rce = -torch.mean(pred*torch.log(labels) + (1-pred)*torch.log(1-labels))

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

class GCELoss(nn.Module):

    def __init__(self, device, q=0.7, k=0.5, trainset_size=50000, width=256, height=256, class_num=1):
        super(GCELoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, class_num, height, width), requires_grad=False).to(device)
        self.device = device
             
    def forward(self, logits, targets, indexes):
        
        p = torch.sigmoid(logits)
        Yp = torch.where(targets == 1, p, 1-p)

        #print(torch.mean(logits))
        #print(torch.mean(p))
        #print(torch.mean(self.weight[indexes]))
        #loss = ((1-(Yp**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = (1-(torch.mean(Yp)**self.q))/self.q
        #loss = torch.mean(loss)
        #print(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = torch.sigmoid(logits)
        Yp = torch.where(targets == 1, p, 1-p)
        Lq = ((1-(Yp**self.q)+1e-6)/self.q)
        Lqk = (1-(self.k**self.q)+1e-6)/self.q
        condition = (Lq<=Lqk).type(torch.cuda.FloatTensor)
        #Lqk = torch.from_numpy((Lq<=Lqk)).type(torch.cuda.FloatTensor)
        #Lqk = torch.unsqueeze(Lqk, 1)
        

        #condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition

class WeightedBCE(nn.Module):
    def __init__(self, device, num_class=3):
        super(WeightedBCE, self).__init__()
        self.device = device
        self.num_class = num_class
             
    def forward(self, outputs, targets):
        weights = torch.zeros(self.num_class)
        weights = weights.to(self.device)
        for target in targets:
            for i in range(3):
                weights[i] += torch.sum(target[i]==0)/(torch.sum(target[i]==1)+1e-6)
        weights = weights/len(targets)
        weights = weights.view(1, 3, 1, 1)
        return nn.BCEWithLogitsLoss(pos_weight=weights)(outputs, targets)

class SoftDiceWithLogitsLoss(nn.Module):
    def __init__(self, nonlinear='sigmoid', smooth=1.0):
        super(SoftDiceWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.nonlinear = nonlinear

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        axes = list(range(2, len(shp_x)))

        if self.nonlinear == 'sigmoid':
            x = robust_sigmoid(x)
        else:
            raise NotImplementedError(self.nonlinear)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)
        
        dc = dc.mean()

        return 1 - dc


class SoftDiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, dice_smooth=1.0):
        """Binary Cross Entropy & Soft Dice Loss 
        
        Seperately return BCEWithLogitsloss and Dice loss.

        BCEWithLogitsloss is more numerically stable than Sigmoid + BCE

        Args:
            bce_kwargs (dict): 
            soft_dice_kwargs (dict):
        """
        super(SoftDiceBCEWithLogitsLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dsc = SoftDiceWithLogitsLoss(nonlinear='sigmoid', smooth=dice_smooth)

    def forward(self, net_output:Tensor, target:Tensor):
        """Compute Binary Cross Entropy & Region Dice Loss

        Args:
            net_output (Tensor): [B, C, ...]
            target (Tensor): [B, C, ...]
        """
        bce_loss = self.bce(net_output, target)
        dsc_loss = self.dsc(net_output, target)

        return bce_loss, dsc_loss

def robust_sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=0.0, max=1.0)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdims=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):    # gt (b, x, y(, z))
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))    # gt (b, 1, x, y(, z))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)     # (b, 1, ...) --> (b, c, ...)

    # shape: (b, c, ...)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn
