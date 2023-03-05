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
    parser = argparse.ArgumentParser(description='SpatialCorrection')
    # Paths
    parser.add_argument('-d', '--dataset', type=str, default=config['dataset'])
    parser.add_argument('--imgs_test', type=str, default=paths['imgs_test'])
    parser.add_argument('--gts_test', type=str, default=paths['gts_test'])
    parser.add_argument('--model_path', type=str, default='./trained/jsrt_example.pth')

    # Visualization
    parser.add_argument('-L', '--log_level', help="Logging levels.",
                        default=config['log_level'], choices=LOG_LEVELS)
    parser.add_argument('--is_progress_bar', type=bool, help="Display progress bar.",
                        default=config['is_progress_bar'], choices=LOG_LEVELS)

    # Training parameters
    parser.add_argument('-cuda', type=int, default=config['cuda'],
                        help='Cuda number')
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
    args = parser.parse_args(args_to_parse)
    return args

def init(args):
    device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")
    return device

def get_input(args):
    if args.dataset == 'Jsrt':
        test_dataset = JSRTDataset(datapath=args.imgs_test, lungpath=args.gts_test[0], heartpath=args.gts_test[1], 
                                claviclepath=args.gts_test[2], mode='test')
        test_loader = torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        return test_loader
    elif args.dataset == 'Brats2020':
        test_dataset = BratsDataset(datapath=args.imgs_test, gtpath=args.gts_test, mode='test')
        test_loader =  torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        return test_loader
    elif args.dataset == 'ISIC2017' or args.dataset == 'Cityscapes':
        test_dataset = ISICDataset(datapath=args.imgs_test, gtpath=args.gts_test, mode='test')
        test_loader =  torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        return test_loader
        
    elif args.dataset == 'LIDC-IDRI':
        test_dataset = LIDCDataset(datapath=args.imgs_test, gtpath=args.gts_test, mode='test')
        test_loader =  torch.utils.data.DataLoader(test_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.num_cpu)
        return test_loader
        
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

    logger.info('1. setup data')
    test_loader = get_input(args)
    
    logger.info('2. setup model')
    if args.dataset == 'Brats2020':
        model = UNet_3D(input_channels=args.input_channel,
                       output_classes=args.num_class,
                       channels_list=[16, 32, 64, 64],
                       ds_layer=4)
    else:
        model = UNet(n_channels=args.input_channel, n_classes=args.num_class, bilinear=True, bias=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    if args.num_gpu>1: model = nn.DataParallel(model, range(args.num_gpu))
    model = model.to(device)

    loss_f = get_loss_f(device=device, **vars(args))

    logger.info('3. start testing')
    test = Evaluator(model, device=device, logger=logger, save_file=False, **vars(args))
    test(test_loader, 0)

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    main(args)
