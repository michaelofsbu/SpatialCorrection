import os, sys
import argparse
from PIL import Image
import random
import numpy as np
import shutil
import cv2
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt as distrans

def get_args(args_to_parse):
    parser = argparse.ArgumentParser(description='Unet')
    # Paths
    parser.add_argument('--gts_root', type=str, default='./Datasets/JSRT/heart_gts_train/',
                                    help='Directory root of groundtruth masks.')
    parser.add_argument('--save_root', type=str, default='./Datasets/JSRT/heart_250_0.8_0.05_0.2/',
                                    help='Directory root for saving generated noisy masks.')
    parser.add_argument('--is3D', type=bool, default=False,
                                    help='Whether the dataset is 3D valume.')
    parser.add_argument('--noisetype', type=str, default='Markov', choices=['Markov', 'DE'],
                                    help='Noise type. Markov: the proposed Markov random noise. DE: dilation and erosion noise.')
    
    # Parameter for Markov random noise
    parser.add_argument('--T', type=int, default=250,
                                    help='Markov process step number.')
    parser.add_argument('--theta1', type=float, default=0.8,
                                    help='Bernoulli parameter controlling preference.')
    parser.add_argument('--theta2', type=float, default=0.05,
                                    help='Bernoulli parameter controlling variance.')
    parser.add_argument('--theta3', type=float, default=0.2,
                                    help='Bernoulli parameter controlling random flipping.')
    
    # Parameter for Dilation and Erosion noise
    parser.add_argument('--range', type=list, default=[9, 11],
                                    help='The range of noise level.')
    args = parser.parse_args(args_to_parse)
    return args

def generate(args):
    '''
        The main function for generating deferent types of noise
    '''
    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)
    gts = sorted([f for f in os.listdir(args.gts_root) if not f.startswith('.')])
    for i in range(len(gts)):
        file = os.path.join(args.gts_root, gts[i])
        save = os.path.join(args.save_root, gts[i])
        # Generate our Markov noise
        if args.noisetype == 'Markov':
            if not args.is3D:
                _MarkovNoise_random(file, save, args.T, args.theta1, args.theta2, args.theta3)
            else:
                _MarkovNoise_random_3D(file, save, args.T, args.theta1, args.theta2, args.theta3)
        # Generate random dilation and erosion noise
        elif args.noisetype == 'DE':
            if not args.is3D:
                _DilateErodeNoise(file, save, args.range)
            else:
                _DilateErodeNoise_3D(file, save, args.range)
        break

def _gradient(img):
    '''
        Calculate gradient of given image, for visualization only
    '''
    kernely = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    edges_xy = cv2.filter2D(img,-1,kernely)
    return edges_xy

def _dice(gt, noise):
    '''
        Calculate dice score for true mask and noise mask, for evaluating noise level
    '''
    eps = 1e-5
    inter = (gt*noise).sum().astype(np.float32)
    return 2*inter/(gt.sum()+noise.sum() + eps).astype(np.float32)

def _DilateErodeNoise(gt_path, save_path, noiseRange):
    '''
        Random dilation an erosion noise, 2D version
    '''
    ntype = np.random.rand()
    gt = np.asarray(Image.open(gt_path))/255
    if ntype > 0.5:
        res = _dilateORerode(gt, 1, noiseRange)
    else:
        res = _dilateORerode(gt, 0, noiseRange)
    print(dice(gt, res))
    gt_noise = Image.fromarray((res*255).astype(np.uint8))
    gt_noise.save(save_path)

def _DilateErodeNoise_3D(gt_path, save_path, noiseRange):
    '''
        Random dilation an erosion noise, 3D version
    '''
    ntype = np.random.rand()
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.float32)
    gt[gt>0] = 1
    noise = gt.copy()
    if ntype > 0.5:
        res = _dilateORerode(noise, 1, noiseRange)
    else:
        res = _dilateORerode(noise, 0, noiseRange)
    print(dice(gt, res))
    noise = sitk.GetImageFromArray((res*255).astype(np.uint8))
    sitk.WriteImage(noise, save_path)

def _dilateORerode(input, ntype, nRange):
    '''
        Dilate or erode given mask, depending on 'ntype'
    '''
    tmp = np.zeros(input.shape)
    nlevel = nRange[0] + (nRange[1] - nRange[0]) * np.random.rand()
    if ntype == 1:
        tmp = (distrans(1-input) < nlevel).astype(np.float32)
    else:
        tmp = 1 - (distrans(input) < nlevel).astype(np.float32)
    return tmp

def _MarkovNoise_random(gtpath, savepath, T, theta1, theta2, theta3):
    '''
        Generate the proposed Markov noise, 2D version
    '''
    gt = np.array(Image.open(gtpath))
    noise = gt.copy()/255
    ps = np.random.rand(T)
    for t in range(T):
        if ps[t] < theta1:
            edge = (distrans(1-noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1]]  = 1 - noise[e[0], e[1]]
        else:
            edge = (distrans(noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1]]  = 1 - noise[e[0], e[1]]
    mask = np.random.rand(noise.shape[0], noise.shape[1])
    noise = np.where(mask>theta3, noise, 1-noise)
    noise = (noise*255).astype(np.uint8)
    noise = cv2.GaussianBlur(noise, (5, 5), 2, 2)
    noise = np.array(noise/255>0.5)
    print('Dice {:.2f}'.format(_dice(gt/255, noise)))
    noise = Image.fromarray((noise*255).astype(np.uint8))
    noise.save(savepath)

def _MarkovNoise_random_3D(gtpath, savepath, T, theta1, theta2, theta3):
    '''
        Generate the proposed Markov noise, 2D version
    '''
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gtpath)).astype(np.float32)
    gt[gt>0] = 1
    noise = gt.copy()
    ps = np.random.rand(T)
    for t in range(T):
        if ps[t] < theta1:
            edge = (distrans(1-noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1], edge[2]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1], e[2]]  = 1 - noise[e[0], e[1], e[2]]
        else:
            edge = (distrans(noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1], edge[2]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1], e[2]]  = 1 - noise[e[0], e[1], e[2]]
    mask = np.random.rand(noise.shape[0], noise.shape[1], noise.shape[2])
    noise = np.where(mask>theta3, noise, 1-noise)
    noise = (noise*255).astype(np.uint8)
    noise = gaussian_filter(noise, 1, mode='nearest')
    noise = np.array(noise/255>0.5)
    print('Dice {:.2f}'.format(_dice(gt, noise)))
    noise = sitk.GetImageFromArray((noise*255).astype(np.uint8))
    sitk.WriteImage(noise, savepath)

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    generate(args)
    
    