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
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/) (you may need to modify `vtoonify_env.yaml` to install PyTorch that matches your own CUDA version following [https://pytorch.org/](https://pytorch.org/)):
```bash
conda env create -f ./environment/vtoonify_env.yaml
```

If you have a problem regarding the cpp extention (fused and upfirdn2d), or no GPU is available, you may refer to [CPU compatible version](./model/stylegan/op_cpu#readme).

<br/>

## (1) Inference for Image/Video Toonification

### Inference Notebook 
<a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>  
To help users get started, we provide a Jupyter notebook found in `./notebooks/inference_playground.ipynb` that allows one to visualize the performance of VToonify.
The notebook will download the necessary pretrained models and run inference on the images found in `./data/`.

### Pre-trained Models

Pre-trained models can be downloaded from [Google Drive]() or [Baidu Cloud]() (access code: sigg):

<table>
    <tr>
        <th>Backbone</th><th>Style</th><th>Model and extrinsic style code link</th>
    </tr>
    <tr>
        <td rowspan="5">DualStyleGAN</td><td>cartoon</td><td>pre-trained VToonify-D and 317 cartoon style codes</td>
    </tr>
    <tr>
        <td>caricature</td><td>pre-trained VToonify-D and 199 caricature style codes</td>
    </tr>
    <tr>
        <td>arcane</td><td>pre-trained VToonify-D and 100 arcane style codes</td>
    </tr> 
    <tr>
        <td>comic</td><td>pre-trained VToonify-D and 101 comic style codes</td>
    </tr>   
    <tr>
        <td>pixar</td><td>pre-trained VToonify-D and 122 pixar style codes</td>
    </tr>   
    <tr>
        <td rowspan="5">Toonify</td><td>cartoon</td><td>pre-trained VToonify-T</td>
    </tr>
    <tr>
        <td>caricature</td><td>pre-trained VToonify-T</td>
    </tr>
    <tr>
        <td>arcane</td><td>pre-trained VToonify-T</td>
    </tr> 
    <tr>
        <td>comic</td><td>pre-trained VToonify-T</td>
    </tr>   
    <tr>
        <td>pixar</td><td>pre-trained VToonify-T</td>
    </tr>   
    <tr>
        <th colspan="2">Supporting model</th><th>Model link</th>
    </tr>
    <tr>
        <td colspan="2">Pixel2style2pixel encoder</td><td><a href="https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view?usp=sharing">encoder.pt</a></td>
    </tr>  
    <tr>
        <td colspan="2">BiSeNet for face parsing</td><td><a href="">faceparsing.pth</a></td>
    </tr>      
</table>

The downloaded models are suggested to be arranged in [this folder structure](./checkpoint/).

### Style Transfer with VToonify-D

Transfer the style of a default Cartoon image onto a default face:
```python
python style_transfer.py --scale_image
```
The results are saved in the folder `./output/`, where `unsplash-rDEOVtE7vOs_input.jpg` is the rescaled input image to fit VToonify and 
`unsplash-rDEOVtE7vOs_vtoonify_d.jpg` is the result. 

Specify the content image and the model, control the style with the following options:
- `--content`: path to the target face image or video
- `--style_id`: the index of the style image (find the mapping between index and the style image [here](https://github.com/williamyang1991/DualStyleGAN/tree/main/doc_images)). 
- `--style_degree` (default: 0.5): adjust the degree of style.
- `--color_transfer`(default: False): perform color transfer if loading a VToonify-Dsdc model.
- `--ckpt`: path of the VToonify-D model. By default, a VToonify-Dsd trained on cartoon style is loaded.
- `--exstyle_path`: path of the extrinsic style code. By default, the cartoon style codes are loaded.
- `--scale_image`: rescale the input image/video to fit VToonify (recommend).
- `--padding` (default: 200, 200, 200, 200): left, right, top, bottom paddings to the face center.

Here is an example of arcane style transfer:
```python
python style_transfer.py --content ./data/unsplash-rDEOVtE7vOs.jpg \
       --scale_image --style_id 0 --style_degree 0.6 \
       --exstyle_path ./checkpoint/arcane/exstyle_code.npy \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt
```

Specify `--video` to perform video toonification:
```python
python style_transfer.py --scale_image --content ./data/YOUR_VIDEO.mp4 --video
```
The above style control options (`--style_id`, `--style_degree`, `--color_transfer`) also work for videos.

### Style Transfer with VToonify-T

Specify `--backbone` as ''toonify'' to load and use a VToonify-T model.
```python
python style_transfer.py --content ./data/unsplash-rDEOVtE7vOs.jpg \
       --scale_image --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_cartoon/vtoonify.pt
```
In VToonify-T, `--style_id`, `--style_degree`, `--color_transfer`, `--exstyle_path` are not used.

As with VToonify-D, specify `--video` to perform video toonification.

<br/>

## (2) Training VToonify

Download the supporting models to the `./checkpoint/` folder and arrange them in [this folder structure](./checkpoint/):

| Model | Description |
| :--- | :--- |
| [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) | StyleGAN model trained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) |
| [encoder.pt](https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view?usp=sharing) | Pixel2style2pixel encoder that embeds FFHQ images into StyleGAN2 Z+ latent code |
| [faceparsing.pth]() | [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) that predicts face parsing maps |
| [directions.npy]() | Editing vectors taken from [LowRankGAN](https://github.com/zhujiapeng/LowRankGAN) for editing face attributes |
| [Toonify](https://github.com/williamyang1991/DualStyleGAN#pretrained-models) \| [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN#pretrained-models) | pre-trained stylegan-based toonification models|

To customize your own style, you may need to train new Toonify/DualStyleGAN model following [here](https://github.com/williamyang1991/DualStyleGAN#3-training-dualstylegan).

### Train VToonify-D

Given the supporting models arranged in the [default folder structure](./checkpoint/), we can simple pre-train the encoder and train the whole VToonify-D by running
```python
# for pre-training the encoder
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train_vtoonify_d.py \
       --stylegan_path DUALSTYLEGAN_PATH --exstyle_path EXSTYLE_CODE_PATH \
       --batch BATCH_SIZE --name SAVING_NAME --pretrain       # + ADDITIONAL STYLE CONTROL OPTIONS
# for training VToonify-D given the pre-trained encoder
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train_vtoonify_d.py \
       --stylegan_path DUALSTYLEGAN_PATH --exstyle_path EXSTYLE_CODE_PATH \
       --batch BATCH_SIZE --name SAVING_NAME                  # + ADDITIONAL STYLE CONTROL OPTIONS
```

### Train VToonify-T

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

The code is mainly developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN).
