import os 
from PIL import Image
import torch
import random
import numpy as np
from scipy import signal
from copy import copy
import shutil
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt as distrans

def get_args(args_to_parse):
    parser = argparse.ArgumentParser(description='Unet')
    # Paths
    parser.add_argument('--gts_root', type=str, default='../Datasets/JSRT/heart_gts_train/'
                                    help='Directory root of groundtruth masks.')
    parser.add_argument('--save_root', type=str, default='../Datasets/JSRT/heart_250_0.8_0.05_0.2/'
                                    help='Directory root for saving generated noisy masks.')
    parser.add_argument('--3D', type=bool, default=False,
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
    return args

def generate(args):
    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)
    gts = sorted([f for f in os.listdir(args.gts_root) if not f.startswith('.')])
    for i in range(len(gts)):
        file = os.path.join(args.gts_root, gts[i])
        save = os.path.join(args.save_root, gts[i])
        if args.noisetype == 'Markov':
            if not args.3D:
                _MarkovNoise_random(file, save, args.T, args.theta1, args.theta2, args.theta3)
            else:
                _MarkovNoise_random_3D(file, save, args.T, args.theta1, args.theta2, args.theta3)
        elif args.noisetype == 'DE':
            if not args.3D:
                _DilateErodeNoise(file, save, args.range)
            else:
                _DilateErodeNoise_3D(file, save, args.range)

def _gradient(img):
    kernely = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    #edges_x = cv2.filter2D(img,-1,kernelx)
    edges_xy = cv2.filter2D(img,-1,kernely)
    return edges_xy

def _DilateErodeNoise(gt_path, save_path, noiseRange):
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
    tmp = np.zeros(input.shape)
    nlevel = nRange[0] + (nRange[1] - nRange[0]) * np.random.rand()
    if ntype == 1:
        tmp = (distrans(1-input) < nlevel).astype(np.float32)
    else:
        tmp = 1 - (distrans(input) < nlevel).astype(np.float32)
    return tmp

def _mix_noise(gtpath, gtpath1, savepath, rate):
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    gts = sorted([f for f in os.listdir(gtpath) if not f.startswith('.')])
    l = int(len(gts)*rate)
    np.random.shuffle(gts)
    for i in range(0,l):
        shutil.copy(gtpath + gts[i], savepath + gts[i])
    for i in range(l, len(gts)):
        shutil.copy(gtpath1 + gts[i], savepath + gts[i])

def _MarkovNoise_random(gtpath, savepath, T, theta1, theta2, theta3):
    gt = np.array(Image.open(gtpath))
    noise = gt.copy()/255
    ps = np.random.rand(T)
    for t in range(T):
        if ps[t] < theta1:
            #edge = ((distrans(1-noise)<1.5)&(distrans(1-noise)>0)).nonzero()
            edge = (distrans(1-noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1]]  = 1 - noise[e[0], e[1]]
            #print('erode')
        else:
            #edge = ((distrans(noise)<1.5)&(distrans(noise)>0)).nonzero()
            edge = (distrans(noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1]]  = 1 - noise[e[0], e[1]]
            # if t%80 == 0:
            #     noise = (noise*255).astype(np.uint8)
            #     noise = cv2.GaussianBlur(noise, (5, 5), 2, 2)
            #     noise = np.array((noise/255)>0.5)
            #print('erode')
    mask = np.random.rand(noise.shape[0], noise.shape[1])
    noise = np.where(mask>theta3, noise, 1-noise)
    #noise = Image.fromarray((noise*255).astype(np.uint8))
    #noise.save(savepath)
    noise = (noise*255).astype(np.uint8)
    noise = cv2.GaussianBlur(noise, (5, 5), 2, 2)
    noise = np.array(noise/255>0.5)
    print(_dice(gt/255, noise))
    noise = Image.fromarray((noise*255).astype(np.uint8))
    noise.save(savepath)

def _dice(gt, pred):
    eps = 1e-5
    #pred = (pred > 0.5).astype(np.float)
    inter = (gt*pred).sum().astype(np.float32)
    return 2*inter/(gt.sum()+pred.sum() + eps).astype(np.float32)

def _MarkovNoise_random_3D(gtpath, savepath, T, theta1, theta2, theta3):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gtpath)).astype(np.float32)
    gt[gt>0] = 1
    noise = gt.copy()
    ps = np.random.rand(T)
    for t in range(T):
        if ps[t] < theta1:
            #edge = ((distrans(1-noise)<1.5)&(distrans(1-noise)>0)).nonzero()
            edge = (distrans(1-noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1], edge[2]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1], e[2]]  = 1 - noise[e[0], e[1], e[2]]
            #print('erode')
        else:
            #edge = ((distrans(noise)<1.5)&(distrans(noise)>0)).nonzero()
            edge = (distrans(noise)==1).nonzero()
            edge = list(zip(edge[0], edge[1], edge[2]))
            prob = np.random.rand(len(edge))
            for i in range(len(edge)):
                e = edge[i]
                if prob[i] < theta2: #flip
                    noise[e[0], e[1], e[2]]  = 1 - noise[e[0], e[1], e[2]]
            # if t%80 == 0:
            #     noise = (noise*255).astype(np.uint8)
            #     noise = cv2.GaussianBlur(noise, (5, 5), 2, 2)
            #     noise = np.array((noise/255)>0.5)
            #print('erode')
    mask = np.random.rand(noise.shape[0], noise.shape[1], noise.shape[2])
    noise = np.where(mask>theta3, noise, 1-noise)
    #noise = Image.fromarray((noise*255).astype(np.uint8))
    #noise.save(savepath)
    noise = (noise*255).astype(np.uint8)
    noise = gaussian_filter(noise, 1, mode='nearest')
    # for i in range(len(noise)):
    #     noise[i] = cv2.GaussianBlur(noise[i], (5, 5), 2, 2)
    noise = np.array(noise/255>0.5)
    print(_dice(gt, noise))
    noise = sitk.GetImageFromArray((noise*255).astype(np.uint8))
    sitk.WriteImage(noise, savepath)
    '''
    edge = gradient(gt)
    plt.figure()
    plt.imshow(noise, cmap='gray')
    plt.scatter(edge.nonzero()[1], edge.nonzero()[0], s=0.5, c='red')
    plt.show()
    #plt.savefig('./markovnoise.png')
    '''

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    generate(args)
    
    