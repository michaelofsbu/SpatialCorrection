import os, sys
import numpy as np
import ast

import torch
import torch.nn as nn
import torch.utils.data
import argparse
import configparser
from random import sample

from net import UNet, UNet_3D
from utils import JSRTDataset, BratsDataset, ISICDataset, Trainer, Evaluator, LIDCDataset
from utils import get_loss_f
from utils import Cleaner, Visualizer

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import logging

CONFIG_FILE = "config.ini"
LOG_LEVELS = list(logging._levelToName.values())
LOG_FILE = 'acc.log'

np.set_printoptions(precision=4,suppress=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_config_section(filenames, section):
    """Return a dictionnary of the section of `.ini` config files. Every value
    int the `.ini` will be litterally evaluated, such that `l=[1,"as"]` actually
    returns a list.
    """
    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    parser.optionxform = str
    files = parser.read(filenames)
    if len(files) == 0:
        raise ValueError("Config files not found: {}".format(filenames))
    dict_session = dict(parser[section])
    dict_session = {k: ast.literal_eval(v) for k, v in dict_session.items()}
    return dict_session

def get_args(args_to_parse):
    config = get_config_section(CONFIG_FILE, 'Common')
    paths = get_config_section(CONFIG_FILE, config['dataset'])

    parser = argparse.ArgumentParser(description='Unet')
    # Paths
    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument('--imgs_tr', type=str, default=paths['imgs_tr'])
    parser.add_argument('--gts_tr', type=str, default=paths['gts_tr'])
    parser.add_argument('--imgs_val', type=str, default=paths['imgs_val'])
    parser.add_argument('--gts_val', type=str, default=paths['gts_val'])
    parser.add_argument('--imgs_test', type=str, default=paths['imgs_test'])
    parser.add_argument('--gts_test', type=str, default=paths['gts_test'])
    parser.add_argument('--gts_true', type=str, default=paths['gts_true'],
                        help='True labels of training data for performance evaluation purpose.')
    parser.add_argument('-d', '--dataset', type=str, default=config['dataset'])
    parser.add_argument('-r', '--cleaned_labels', type=str, default=config['cleaned_root'],
                        help='The root for saving cleaned labels.')
    parser.add_argument('--val_size', type=int, default=config['val_size'],
                        help='Validation Size.')

    # Visualization
    parser.add_argument('-L', '--log_level', help="Logging levels.",
                        default=config['log_level'], choices=LOG_LEVELS)
    parser.add_argument('--is_progress_bar', type=bool, help="Display progress bar.",
                        default=config['is_progress_bar'], choices=LOG_LEVELS)

    # Training parameters
    parser.add_argument('-cuda', type=int, default=config['cuda'],
                        help='Cuda number')
    parser.add_argument('-lr', type=float, default=config['lr'],
                        help='Learning rate')
    parser.add_argument('-e','--max_epoch', type=int, default=config['epochs'],
                        help='Total number of epoch')
    parser.add_argument('--epoch_save', type=int, default=config['checkpoint_every'],
                        help='Number of epoch to save')
    parser.add_argument('-g','--num_gpu', type=int,  default=config['num_gpu'],
                        help='Number of gpu')
    parser.add_argument('-c','--num_cpu', type=int,  default=config['num_cpu'],
                        help='Number of cpu')
    parser.add_argument('-b','--batch_size', type=int,  default=config['batch_size'],
                        help='Batch size')
    parser.add_argument('-ch','--input_channel', type=int,  default=config['input_channel'],
                        help='Input Channel')
    parser.add_argument('-cl','--num_class', type=int,  default=config['num_class'],
                        help='Class number')
    parser.add_argument('--model_num', type=int,  default=config['model_num'],
                        help='Model number')

    # Label clean parameters
    parser.add_argument('-s', '--sigma', type=float, default=config['sigma'],
                        help='Parameter in label cleaning')
    parser.add_argument('-le', '--labelclean_ever', type=int, default=config['labelclean_every'],
                        help='Number of epoch to clean training label')
    parser.add_argument('--max_iter', type=int, default=config['max_iter'],
                        help='Max number of iterations to clean training label')
    parser.add_argument('--startover', type=bool, default=config['startover'],
                        help='Whether to train from the start after label cleaning.')
    parser.add_argument('--save_labels', type=bool, default=config['save_labels'],
                        help='Whether to save cleaned labels. No use')
    parser.add_argument('--show_clean_performance', type=bool, default=config['show_clean_performance'],
                        help='Whether to show the accuracy of cleaned labels. Requires true trianing labels available.')

    args = parser.parse_args(args_to_parse)
    args.output = os.path.join('./checkpoints', args.dataset, args.name)
    args.cleaned = os.path.join(args.cleaned_labels, args.dataset, args.name)
    return args

def init(args):
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")
    return device

def get_input(args):
    if args.dataset == 'Jsrt':
        train_dataset = JSRTDataset(datapath=args.imgs_tr, lungpath=args.gts_tr[0], heartpath=args.gts_tr[1], 
                                claviclepath=args.gts_tr[2], mode='train')
        valid_dataset = JSRTDataset(datapath=args.imgs_val, lungpath=args.gts_val[0], heartpath=args.gts_val[1], 
                                claviclepath=args.gts_val[2], mode='val', size = args.val_size)
        test_dataset = JSRTDataset(datapath=args.imgs_test, lungpath=args.gts_test[0], heartpath=args.gts_test[1], 
                                claviclepath=args.gts_test[2], mode='test')
        train_loader =  torch.utils.data.DataLoader(train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_cpu)
        valid_loader =  torch.utils.data.DataLoader(valid_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        if args.show_clean_performance == True:
            true_dataset = JSRTDataset(datapath=args.imgs_tr, lungpath=args.gts_true[0], heartpath=args.gts_true[1], 
                                claviclepath=args.gts_true[2], mode='clean')
            true_loader = torch.utils.data.DataLoader(true_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
            return train_loader, valid_loader, test_loader, true_loader
        else:
            return train_loader, valid_loader, test_loader
    elif args.dataset == 'Brats2020':
        train_dataset = BratsDataset(datapath=args.imgs_tr, gtpath=args.gts_tr, mode='train')
        valid_dataset = BratsDataset(datapath=args.imgs_val, gtpath=args.gts_val, mode='val')
        test_dataset = BratsDataset(datapath=args.imgs_test, gtpath=args.gts_test, mode='test')
        train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_cpu)
        valid_loader =  torch.utils.data.DataLoader(valid_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        test_loader =  torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        if args.show_clean_performance == True:
            true_dataset = BratsDataset(datapath=args.imgs_tr, gtpath=args.gts_true, mode='clean')
            true_loader = torch.utils.data.DataLoader(true_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
            return train_loader, valid_loader, test_loader, true_loader
        else:
            return train_loader, valid_loader, test_loader
    elif args.dataset == 'ISIC2017':
        train_dataset = ISICDataset(datapath=args.imgs_tr, gtpath=args.gts_tr, mode='train')
        valid_dataset = ISICDataset(datapath=args.imgs_val, gtpath=args.gts_val, mode='val')
        test_dataset = ISICDataset(datapath=args.imgs_test, gtpath=args.gts_test, mode='test')
        train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_cpu)
        valid_loader =  torch.utils.data.DataLoader(valid_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        test_loader =  torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        if args.show_clean_performance == True:
            true_dataset = ISICDataset(datapath=args.imgs_tr, gtpath=args.gts_true, mode='clean')
            true_loader = torch.utils.data.DataLoader(true_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
            return train_loader, valid_loader, test_loader, true_loader
        else:
            return train_loader, valid_loader, test_loader
        
    elif args.dataset == 'LIDC-IDRI':
        train_dataset = LIDCDataset(datapath=args.imgs_tr, gtpath=args.gts_tr, mode='train')
        valid_dataset = LIDCDataset(datapath=args.imgs_val, gtpath=args.gts_val, mode='val')
        test_dataset = LIDCDataset(datapath=args.imgs_test, gtpath=args.gts_test, mode='test')
        train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_cpu)
        valid_loader =  torch.utils.data.DataLoader(valid_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        test_loader =  torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        if args.show_clean_performance == True:
            true_dataset = LIDCDataset(datapath=args.imgs_tr, gtpath=args.gts_true, mode='clean')
            true_loader = torch.utils.data.DataLoader(true_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
            return train_loader, valid_loader, test_loader, true_loader
        else:
            return train_loader, valid_loader, test_loader

def weight_init(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        
def main(args):
    device = init(args)
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logger.propagate = False
    logger.info('Root directory for model saving: {}'.format(args.output))

    save_logger = logging.getLogger('SAVE')
    hdlr = logging.FileHandler(os.path.join(args.output, LOG_FILE), mode='w')
    hdlr.setFormatter(formatter)
    save_logger.setLevel(logging.INFO)
    save_logger.addHandler(hdlr)

    logger.info('0. initial setup') 
    writer = SummaryWriter('./runs/'+args.output.split('/')[-1])
    save_logger.info('[lr: {:.5f}], [batch_size: {:d}]'.format(args.lr, args.batch_size))

    logger.info('1. setup data')
    #if args.dataset == 'Jsrt':
    train_loader, eval_loader, test_loader, true_loader = get_input(args)
    # elif args.dataset == 'Brats2020':
    #     train_loader, eval_loader, test_loader, true_loader = get_input(args)
    
    logger.info('2. setup model')
    if args.dataset == 'Brats2020':
        model = UNet_3D(input_channels=args.input_channel,
                       output_classes=args.num_class,
                       channels_list=[16, 32, 64, 64],
                       #channels_list=[16, 32, 64, 128],
                    ds_layer=4)
    else:
        model = UNet(n_channels=args.input_channel, n_classes=args.num_class, bilinear=True, bias=True)
            
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu))
    model = model.to(device)
    #model.load_state_dict(torch.load(os.path.join('./checkpoints/LIDC-IDRI/noise/model1.pth'), map_location=device))
    #torch.save(model.state_dict(), os.path.join('./checkpoints', args.dataset, 'init.pth'))
    loss_f = get_loss_f(device=device, **vars(args))

    logger.info('3. setup optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    logger.info('4. start training')
    trainer = Trainer(model, optimizer, loss_f, scheduler, writer, device, logger, **vars(args))
    evaluator = Evaluator(model, writer, device=device, logger=logger, **vars(args))
    cleaner = Cleaner(model, true_loader, logger=save_logger, device=device, **vars(args))
    test = Evaluator(model, writer, device=device, logger=save_logger, save_file=True, **vars(args))
    vis = Visualizer(len(true_loader), logger=save_logger, dataset=args.dataset)
    vis.append_loader(train_loader)
    vis.append_loader(true_loader)
    vis.append_prediction(None, eval_loader, device)
    dsc_list = []
    for i in range(args.model_num):
        optimizer.param_groups[0]['lr'] = args.lr
        logger.info('Training model {:d}'.format(i+1))
        for e in range(1, args.max_epoch+1):
            trainer(train_loader, e, args.max_epoch)
            evaluator(eval_loader, e)
            #scheduler.step()
            iter = int(e/args.labelclean_ever)
            if e % args.labelclean_ever == 0 and iter <= args.max_iter:
                if i == 0:
                    vis.append_prediction(model, eval_loader, device)
                test(test_loader, e)
                cleaner(train_loader, eval_loader, iter, args.sigma)
                optimizer.param_groups[0]['lr'] = args.lr
                if i == 0:
                    vis.append_loader(train_loader)

        if i == 0:
            vis.append_prediction(model, eval_loader, device)
            vis.save(args.output)
        dsc = test(test_loader, e)
        dsc_list.append(dsc)
        torch.save(model.state_dict(), os.path.join(args.output, ('model{:d}.pth'.format(i+1))))
        model.apply(weight_init)
        #model.load_state_dict(torch.load(os.path.join('./checkpoints', args.dataset, 'init.pth'), map_location=device))
        train_loader, eval_loader, test_loader, true_loader = get_input(args)
    logger.info('5. finish training')

    logger.info('6. Cross Evaluation')
    test.cross_eval(dsc_list, dataset=args.dataset)
    writer.close()

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    main(args)
