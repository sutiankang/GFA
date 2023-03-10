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

### Prepare Optical Flow
Please following the instruction of [RAFT](https://github.com/princeton-vl/RAFT) to prepare the optical flow. Note that this repository is from [RAFT: Recurrent All Pairs Field Transforms for Optical Flow (ECCV 2020)](https://arxiv.org/pdf/2003.12039.pdf).

## Test

- The test datasets can be downloaded from [DAVIS-2016](https://davischallenge.org/davis2017/code.html), [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html), [Youtube-objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects/), [DAVSOD](https://github.com/DengPingFan/DAVSOD), [SegTrack-V2](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html), [MCL](https://github.com/DengPingFan/DAVSOD), [Robotic Instrument](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/). Note that we use ```DAVSOD``` repository to replace ```MCL``` due to it provides download way.

The structure of datasets is as follows:
```
|—— Datasets 
  |—— DAVIS-2016
    |—— train
      |—— images
        |—— 00000.jpg
        |—— 00001.jpg
        |—— ...
      |—— flows
        |—— 00000.jpg
        |—— 00001.jpg
        |—— ...
      |—— labels 
        |—— 00000.png
        |—— 00001.png
        |—— ...
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
    ...
```

-   We can produce segmentation results in ```test.py```.
```
python test.py --img_size 512 --save_dir runs/GFA/test --use_flip -v DAVIS-2016 --model segformer_b5 --is_binary --weights /your/final_weight/path
```

## Final weight

- The final weight can be downloaded from [GFA](https://anonymfile.com/VxB6l/gfa.pth).

## Evaluation Metrics

- We use the standard UVOS evaluation toolbox from [DAVIS-2016 benchmark](https://github.com/davisvideochallenge/davis-matlab/tree/davis-2016) and VSOD evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD). Note that the two toolboxes are from official repositories. 
