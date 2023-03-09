# Generalizable Fourier Augmentation for Video Object Segmentation

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.10.1 with four GeForce RTX 2080Ti GPUs with 11GB Memory.
- Python 3.8
```
conda create -n mp-vos python=3.8
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
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): We use all the data in the val subset of FBMS. Note that this only used to evaluate results.

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
The pre-trained backbone can be downloaded from [Segformer backbone]() and revise ```utils.py/create_model```.

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

-   We can produce segmentation results in ```test.py```.
```
python test.py --img_size 512 --save_dir runs/GFA/test --use_flip -v DAVIS-2016 --model segformer_b5 --is_binary --weights /your/final_weight/path
```

## Final weight

- The final weight can be downloaded from [GFA](https://anonymfile.com/YPeR8/gfa.pth).

## Evaluation Metrics

- We use the standard UVOS evaluation toolbox from [DAVIS-2016 benchmark](https://github.com/davisvideochallenge/davis-matlab/tree/davis-2016) and VSOD evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD). Note that the two toolboxes are from official repositories. 
- The IIW metric can refer to [IIW](https://github.com/RyanWangZf/PAC-Bayes-IB). Note that this also belongs to  official repository.
