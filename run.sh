# Brats2020
#python main.py noise3 -cuda 3 -e 60 -le 20 --max_iter 2 -lr 0.001 -b 2 -ch 1 -cl 1 -s 0.8

# Jsrt
#python main.py noise2 --dataset Jsrt -cuda 6 -e 100 -le 50 --max_iter 1 -lr 0.01 -b 20 -ch 1 -cl 3 -s 0.8

#ISIC2017
#python main.py noise2 -cuda 6 -e 100 -le 50 --max_iter 1 -lr 0.005 -b 10 -ch 3 -cl 1 -s 0.8

#ISIC2017 noise 3/4
#python main.py noise4 -cuda 4 -e 80 -le 25 --max_iter 2 -lr 0.005 -b 10 -ch 3 -cl 1 -s 0.3

#Cityscapes noise
#python main.py noise -cuda 2 -e 80 -le 25 --max_iter 2 -lr 0.005 -b 10 -ch 3 -cl 1 -s 0.3

# Jsrt noise 3
#python main.py val_1 --dataset Jsrt -cuda 6 -e 50 -le 25 --max_iter 1 -lr 0.01 -b 20 -ch 1 -cl 3 -s 0.7 --val_size 1

# LIDC
#python main.py pretrain --dataset LIDC-IDRI -cuda 3 -e 40 -le 60 --max_iter 0 -lr 0.05 -b 64 -ch 1 -cl 1 -s 0.8

# LIDC
python main.py noisel_true_sce_lr_01 --dataset LIDC-IDRI -cuda 1 -e 30 -le 15 --max_iter 1 -lr 0.1 -b 64 -ch 1 -cl 1 -s 1