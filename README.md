# Learning to Segment from Noisy Annotations: A Spatial Correction Approach

This is the official repository for the ICLR 2023 paper **[Learning to Segment from Noisy Annotations: A Spatial Correction Approach](https://openreview.net/forum?id=Qc_OopMEBnC&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions))**. This paper proposed a Markov model for simulating segmentation label noise, and a Spatial Correction method to combat such noise. The outline of this repository is given as follows.

* [**Requirements**](#requirements)
* [**Noisy Label Settings**](#noisy-label-settings)
* [**Test with Trained Models**](#test-with-trained-models)
* [**Train on Your Own**](#train-on-your-own)
* [**Citing**](#citing)

## Requirements

Our code has been tested on Ubuntu 20.04.5 LTS with CUDA 11.4 (Driver Version 470.161.03), Python 3.9.7, and PyTorch v1.10.2. (Add .requirements)

## Noisy Label Settings

The proposed method is compared with SOTAs in a wide range of noisy settings, including both synthetic and real-world noise.

### Synthetic Noise

Previous works used random dilation and/or erosion noise to simulate noisy annotations, while we propose a Markov process to model such noise. The comparison is as follows,

|![fig1](figs/noisetype.png)|
|:---|
|Comparison of different types of noise. Blue lines are true segmentation boundaries, and red lines are noisy boundaries (after removing random flipping noise). $S_E$ and $S_S$ are generated by the proposed Markov mode.|

The Markov noise generation follows Definition 1 in our paper. There are four parameters controlling this process, $T$ the number of steps, $\theta_1$ the noise preference, $\theta_2$ noise variance, $\theta_3$ random flipping noise rate. To generate the proposed Markov noise for you own dataset, run
```
python noise_generator.py --gts_root your/gt/root/ --save_root your/save/root/
```
To generate random dilation and erosion noise, add `--noisetype DE` to the above command.\
Other arguments can be set accordingly with detailed descriptions inside the function. To generate noisy labels used in Table 1 in the paper, refer to the following settings.

|   ID   |   Dataset   | `--is3D` | `--noisetype` |                 `--range`                  |                `--T`                 | `--theta1` |               `--theta2`                | `--theta3` |
| :----: | :---------: | :------: | :-----------: | :----------------------------------------: | :----------------------------------: | :--------: | :-------------------------------------: | :--------: |
| **J1** |   *JSRT*    |  False   |    Markov     |                     --                     | Lung: 180, Heart: 180, Clavicle: 100 |    0.8     | Lung: 0.05, Heart: 0.05, Clavicle: 0.02 |    0.2     |
| **J2** |   *JSRT*    |  False   |    Markov     |                     --                     | Lung: 180, Heart: 180, Clavicle: 100 |    0.2     | Lung: 0.05, Heart: 0.05, Clavicle: 0.02 |    0.2     |
| **J3** |   *JSRT*    |  False   |      DE       | Lung: [7,9], Heart: [7,9], Clavicle: [3,5] |                  --                  |     --     |                   --                    |     --     |
| **B1** | *Brats2020* |   True   |    Markov     |                     --                     |                  80                  |    0.7     |                  0.05                   |     0      |
| **B2** | *Brats2020* |   True   |    Markov     |                     --                     |                  80                  |    0.3     |                  0.05                   |     0      |
| **B3** | *Brats2020* |   True   |      DE       |                   [3,5]                    |                  --                  |     --     |                   --                    |     --     |
| **I1** | *ISIC2017*  |  False   |    Markov     |                     --                     |                 200                  |    0.2     |                  0.05                   |    0.2     |
| **I2** | *ISIC2017*  |  False   |    Markov     |                     --                     |                 200                  |    0.8     |                  0.05                   |    0.2     |
| **I3** | *ISIC2017*  |  False   |      DE       |                   [7,9]                    |                  --                  |     --     |                   --                    |     --     |

### Real-world Noise

We include Cityscapes and LIDC-IDRI datasets for real-world label noise settings. Detailed information can be found in the paper. Our selected Cityscapes dataset can be downloaed here soon.

## Test with Trained models

We provide the *JSRT* dataset with noisy setting **J1** in `./Datasets` and a trained model with the proposed Spatial Correction method in `./trained`. If you use *JSRT* dataset in your work, please cite their original publications. More trained models will be uploaded later.

For testing on the provided model, simply run `python test.py`. To test other models, change the paths in `test.py` accordingly and run the same command.

## Train on Your Own

**Train with datasets in the paper.**

**Train with your own dataset.**

## Citing

If you find this code helpful, please consider citing as

```
@inproceedings{yao2023spatialcorrection,
   title={Learning to Segment from Noisy Annotations: A Spatial Correction Approach},
   author={Yao, Jiachen and Zhang, Yikai and Zheng, Songzhu and Goswami, Mayank and Prasanna, Prateek and Chen, Chao},
   booktitle={International Conference on Learning Representations},
   year={2023}
}
```



