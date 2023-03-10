# Generalizable Fourier Augmentation for Video Object Segmentation

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.10.1 with four GeForce RTX 2080Ti GPUs with 11GB Memory.
- Python 3.8
```
conda create -n gfa python=3.8
```
Other minor Python modules can be installed by running
```
pip install -r requirements.txt
```

## Train

### Download Datasets
In the paper, we use the following three public available dataset for training. Here are some steps to prepare the data:
- [DAVIS-2016](https://davischallenge.org/davis2017/code.html): We use all the data in the train subset of DAVIS-16. However, please download DAVIS-17 dataset, it will automatically choose the subset of DAVIS-16 for training.
- [YouTubeVOS-2018](https://youtube-vos.org/dataset/): We sample the training data every 10 frames as labeled data in YoutubeVOS-2018. You can sample any number of frames to train the model by modifying parameter ```--stride```.
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): We only employ FBMS datasets for evaluating without any training.

Note that these datasets are all public.

### Prepare Optical Flow
Please following the instruction of [RAFT](https://github.com/princeton-vl/RAFT) to prepare the optical flow. Note that this repository is from [RAFT: Recurrent All Pairs Field Transforms for Optical Flow (ECCV 2020)](https://arxiv.org/pdf/2003.12039.pdf).

The structure of datasets is as follows:
```
|—— Datasets
  |—— YouTubeVOS-2018
    |—— train
      |—— images
        |—— 00000.jpg
        |—— 00005.jpg
        |—— ...
      |—— flows
        |—— 00000.jpg
        |—— 00005.jpg
        |—— ...
      |—— labels
        |—— 00000.png
        |—— 00005.png
        |—— ...
    |—— val
      |—— images
      |—— flows
      |—— labels    
  |—— DAVIS-2016
    |—— train
      |—— images
      |—— flows
      |—— labels    
    |—— val
      |—— images
      |—— flows
      |—— labels
  |—— FBMS
    |—— train
      |—— images
      |—— flows
      |—— labels    
    |—— val
      |—— images
      |—— flows
      |—— labels    
```

### Prepare pretrained backbone
The pre-trained backbone can be downloaded from [Segformer-b0 backbone](https://anonymfile.com/8p2KA/segformer-b0.zip), [Segformer-b1 backbone](https://anonymfile.com/rNbRx/segformer-b1.zip), [Segformer-b2 backbone](https://anonymfile.com/k0a9Q/segformer-b2.zip), [Segformer-b3 backbone](https://anonymfile.com/4YE4O/segformer-b3.zip), [Segformer-b4 backbone](https://anonymfile.com/392mK/segformer-b4.zip), [Segformer-b5 backbone](https://anonymfile.com/1J2rY/segformer-b5.zip),  and revise the path of pretrained backbones in ```utils/utils.py/create_model```.

### Train
- First, train the model using the YouTubeVOS-2018, DAVIS-2016 datasets.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 train.py --epochs 30 -t YouTubeVOS-2018 DAVIS-2016 -v DAVIS-2016 FBMS --batch_size 4 --amp --sync_bn --warmup_epochs 0 --lr 1e-4 --epochs 30 --model segformer_b5 --pretrained --img_size 512  --uncertainty_probability 0.5 --tensorboard --stride 10 --data_dir /your/data/path --experiment runs/GFA
```
- Second, finetune the model using the DAVIS-2016 dataset.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 train.py --epochs 50 -t YouTubeVOS-2018 DAVIS-2016 -v DAVIS-2016 FBMS --batch_size 4 --amp --sync_bn --warmup_epochs 0 --lr 1e-4 --epochs 30 --model segformer_b5 --pretrained --img_size 512  --uncertainty_probability 0.5 --tensorboard --stride 10 --data_dir /your/data/path --finetune /your/first_stage/path --experiment runs/GFA
```


## Test

- The test datasets can be downloaded from [DAVIS-2016](https://davischallenge.org/davis2017/code.html), [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html), [Youtube-objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects/), [DAVSOD](https://github.com/DengPingFan/DAVSOD), [ViSal](https://github.com/DengPingFan/DAVSOD), [SegTrack-V2](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html), [MCL](https://github.com/DengPingFan/DAVSOD), [Robotic Instrument](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/). Note that we use ```DAVSOD``` repository to replace ```ViSal``` and ```MCL``` due to it provides these two dataset download way. Besides, the stucture must be the same as training example.

-   We can produce segmentation results in ```test.py```.
```
python test.py --img_size 512 --save_dir runs/GFA/test --use_flip -v DAVIS-2016 --model segformer_b5 --is_binary --weights /your/final_weight/path
```

## Final weight

- The final weight can be downloaded from [GFA](https://anonymfile.com/VxB6l/gfa.pth).

## Evaluation Metrics

- We use the standard UVOS evaluation toolbox from [DAVIS-2016 benchmark](https://github.com/davisvideochallenge/davis-matlab/tree/davis-2016) and VSOD evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD). Note that the two toolboxes are from official repositories. 
- The IIW metric can refer to [IIW](https://github.com/RyanWangZf/PAC-Bayes-IB). Note that this also belongs to  official repository.
