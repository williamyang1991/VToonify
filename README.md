# VToonify - Official PyTorch Implementation

This repository provides the official PyTorch implementation for the following paper:

**VToonify: Controllable High-Resolution Portrait Video Style Transfer**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
In ACM TOG (Proceedings of SIGGRAPH Asia), 2022.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/vtoonify/) | [**Paper**]() | [**Supplementary Video**]()<br>

<a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> ![visitors](https://visitor-badge.glitch.me/badge?page_id=williamyang1991/VToonify)
> **Abstract:** *Generating high-quality artistic portrait videos is an important and desirable task in computer graphics and vision.
Although a series of successful portrait image toonification models built upon the powerful StyleGAN have been proposed,
these image-oriented methods have obvious limitations when applied to videos, such as the fixed frame size, the requirement of face alignment, missing non-facial details and temporal inconsistency.
In this work, we investigate the challenging controllable high-resolution portrait video style transfer by introducing a novel **VToonify** framework.
Specifically, VToonify leverages the mid- and high-resolution layers of StyleGAN to render high-quality artistic portraits based on the multi-scale content features extracted by an encoder to better preserve the frame details. The resulting fully convolutional architecture accepts non-aligned faces in videos of variable size as input, contributing to complete face regions with natural motions in the output.
Our framework is compatible with existing StyleGAN-based image toonification models to extend them to video toonification, and inherits appealing features of these models for flexible style control on color and intensity.
This work presents two instantiations of VToonify built upon Toonify and DualStyleGAN for collection-based and exemplar-based portrait video style transfer, respectively.
Extensive experimental results demonstrate the effectiveness of our proposed VToonify framework over existing methods in generating high-quality and temporally-coherent artistic portrait videos with flexible style controls.*

**Features**:<br> 
**High-Resolution Video** (>1024, support unaligned faces) | **Data-Friendly** (no real training data) | **Style Control**

## Updates

- [09/2022] This website is created.


## Installation
**Clone this repo:**
```bash
git clone https://github.com/williamyang1991/VToonify.git
cd VToonify
```
**Dependencies:**

We have tested on:
- CUDA 10.1
- PyTorch 1.7.0
- Pillow 8.3.1; Matplotlib 3.3.4; opencv-python 4.5.3; Faiss 1.7.1; tqdm 4.61.2; Ninja 1.10.2

All dependencies for defining the environment are provided in `environment/vtoonify_env.yaml`.
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/):
```bash
conda env create -f ./environment/vtoonify_env.yaml
```
We use CUDA 10.1 and PyTorch 1.7.1 (corresponding to [Line 22](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/vtoonify_env.yaml#L22), [Line 25](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/vtoonify_env.yaml#L25), [Line 26](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/vtoonify_env.yaml#L26) of `vtoonify_env.yaml`). Please install PyTorch that matches your own CUDA version following [https://pytorch.org/](https://pytorch.org/).

If you have a problem regarding the cpp extention (fused and upfirdn2d), or no GPU is available, you may refer to [CPU compatible version](./model/stylegan/op_cpu#readme).


## (1) Inference for Image/Video Toonification

## (2) Training VToonify

Download the supporting models to the `./checkpoint/` folder:

| Model | Description |
| :--- | :--- |
| [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) | StyleGAN model trained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch). |
| [encoder.pt](https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view?usp=sharing) | Pixel2style2pixel encoder that embeds FFHQ images into StyleGAN2 Z+ latent code |
| [faceparsing.pt]() | BiSeNet that predicts face parsing maps |
| [derections.npy]() | Editing vectors taken from [LowRankGAN](https://github.com/zhujiapeng/LowRankGAN) for editing face attributes |
| [Toonify](https://github.com/williamyang1991/DualStyleGAN#pretrained-models) \| [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN#pretrained-models) | pre-trained stylegan-based toonify models |

The saved checkpoints are suggested to follow the [folder structure](./checkpoint/).

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{yang2022Vtoonify,
  title={VToonify: Controllable High-Resolution Portrait Video Style Transfer},
  author={Yang, Shuai and Jiang, Liming and Liu, Ziwei and Loy, Chen Change},
  journal={ACM Transactions on Graphics (TOG)},
  year={2022}
}
```

## Acknowledgments

The code is mainly developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel).
