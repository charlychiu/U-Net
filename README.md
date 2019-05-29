# U-Net with Keras

<p align="center">
<a href="https://arxiv.org/abs/1505.04597"><img src="https://img.shields.io/badge/arXiv-1505.04597-orange.svg" alt="Paper"></a>

</p>

Keras implementation of U-Net, with simple annotation

> U-Net: Convolutional Networks for Biomedical Image Segmentation
> Paper URL : https://arxiv.org/abs/1505.04597

## Dataset
[ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/home)    
[Cell Tracking Challenge](http://celltrackingchallenge.net)  
    DIC-C2DH-HeLa (HeLa cells on a flat glass)  
    PhC-C2DH-U373 (Glioblastoma-astrocytoma U373 cells on a polyacrylamide substrate)
    
## Preprocessed

### Original *512x512*
![training-data-0](images/original.jpg)

### Overlap-tile strategy *696x696*
![training-data-0](images/overlap-tile.jpg)


## Paper 

### Model Structure
![training-data-0](images/u-net-architecture.png)

### Overlap-tile strategy
![training-data-0](images/u-net-overlap-tile.png)