from tqdm import tqdm
import torch
import logging
import os


class Trainer():
    def __init__(self, model, optimizer, loss_f, 
                 scheduler, writer,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 **kwargs) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.logger = logger
        self.is_progress_bar = kwargs['is_progress_bar']
        self.dataset = kwargs['dataset']
        self.logger.info("Training Device: {}".format(self.device))
    
    def __call__(self, train_loader,
                 epoch, max_epoch = 100) -> None:
        kwargs = dict(desc=f'Epoch {epoch}/{max_epoch}', leave=False,
                      disable=not self.is_progress_bar)
        train_loss = 0.0
        self.model.train()
        if self.dataset == 'Brats2020':
            with tqdm(total=len(train_loader.dataset), **kwargs) as pbar:
                for iteration, batch in enumerate(train_loader):
                    bce_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                    dsc_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                    images, targets  = batch['image'].to(self.device), batch['mask'].to(self.device)
                    targets = targets.type(torch.float32) #if args.num_class == 1 else target.type(torch.long)
                    outputs = self.model(images)
                    # calc loss weighting factor, works for both w/ or w/o deep supervision
                    # weights as numpy array will make the compute graph empty when using amp
                    weights = torch.pow(0.5, torch.arange(len(outputs)))
                    weights /= weights.sum()
                    #calc losses
                    for j in range(len(outputs)):
                        bce, dsc = self.loss_f(outputs[j], targets)
                        bce_loss += weights[j] * bce
                        dsc_loss += weights[j] * dsc
                    #import pdb; pdb.set_trace()
                    #bce_loss, dsc_loss = self.loss_f(outputs[0], targets)
                    loss = bce_loss + dsc_loss
                    #loss = self.loss_f(outputs[0], targets)
                    #loss = criterion(outputs, labels, cm)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.update(images.shape[0])
        else:
            with tqdm(total=len(train_loader.dataset), **kwargs) as pbar:
                for iteration, batch in enumerate(train_loader):
                    image, target  = batch['image'].to(self.device), batch['mask'].to(self.device)
                    target = target.type(torch.float32) #if args.num_class == 1 else target.type(torch.long)
                    inputs = image
                    labels = target
                    outputs = self.model(inputs)
                    loss = self.loss_f(outputs, labels)
                    #loss = criterion(outputs, labels, cm)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.update(image.shape[0])
        #self.scheduler.step()
        train_loss = train_loss/(iteration+1)
        self.logger.info("[epoch %d] train_loss=%0.4f lr=%.5f" % (epoch, \
                    train_loss, self.optimizer.param_groups[0]['lr']))
        self.writer.add_scalar('train_loss', train_loss, epoch)