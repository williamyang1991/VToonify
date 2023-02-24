# VToonify - Official PyTorch Implementation

https://user-images.githubusercontent.com/18130694/189483939-0fc4a358-fb34-43cc-811a-b22adb820d57.mp4

This repository provides the official PyTorch implementation for the following paper:

**VToonify: Controllable High-Resolution Portrait Video Style Transfer**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
In ACM TOG (Proceedings of SIGGRAPH Asia), 2022.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/vtoonify/) | [**Paper**](https://arxiv.org/abs/2209.11224) | [**Supplementary Video**](https://youtu.be/0_OmVhDgYuY) | [**Input Data and Video Results**](https://drive.google.com/file/d/1A2gC2PW1ZmU6VWQRvMN98njqRxfLjqbk/view?usp=sharing) <br>

<a href="http://colab.research.google.com/github/williamyang1991/VToonify/blob/master/notebooks/inference_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PKUWilliamYang/VToonify)
[![Deque](https://img.shields.io/badge/Deque-Notebook-blue)](https://deque.ai/experience/notebook/dbabc82a-ace4-11ed-84b5-0242ac110002)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=williamyang1991/VToonify)
<!--![visitors](https://visitor-badge.glitch.me/badge?page_id=williamyang1991/VToonify)-->


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

![overview](https://user-images.githubusercontent.com/18130694/189509940-91c5e1e2-83a8-491e-962e-64775e56d7f6.jpg)


## Updates

- [02/2023] Integrated to [Deque Notebook](https://deque.ai/experience/notebook/dbabc82a-ace4-11ed-84b5-0242ac110002).
- [10/2022] Integrate [Gradio](https://gradio.app/) interface into [Colab notebook](http://colab.research.google.com/github/williamyang1991/VToonify/blob/master/notebooks/inference_playground.ipynb). Enjoy the web demo!
- [10/2022] Integrated to 🤗 [Hugging Face](https://huggingface.co/spaces/PKUWilliamYang/VToonify). Enjoy the web demo!
- [09/2022] Input videos and video results are released.
- [09/2022] Paper is released.
- [09/2022] Code is released.
- [09/2022] This website is created.

## Web Demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PKUWilliamYang/VToonify)


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

☞ Install on Windows: https://github.com/williamyang1991/VToonify/issues/50#issuecomment-1443061101 and https://github.com/williamyang1991/VToonify/issues/38#issuecomment-1442146800

☞ If you have a problem regarding the cpp extention (fused and upfirdn2d), or no GPU is available, you may refer to [CPU compatible version](./model/stylegan/op_cpu#readme).

<br/>

## (1) Inference for Image/Video Toonification

### Inference Notebook 
<a href="http://colab.research.google.com/github/williamyang1991/VToonify/blob/master/notebooks/inference_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
[![Deque](https://img.shields.io/badge/Deque-Notebook-blue)](https://deque.ai/experience/notebook/dbabc82a-ace4-11ed-84b5-0242ac110002)

To help users get started, we provide a Jupyter notebook found in `./notebooks/inference_playground.ipynb` that allows one to visualize the performance of VToonify.
The notebook will download the necessary pretrained models and run inference on the images found in `./data/`.

### Pre-trained Models

Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Nmbz9zBM78I1nRVokhHLuBKHleDmjDxv?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1Io3PKNV1wD7ttxaVz-MPww?pwd=sigg) (access code: sigg) or [Hugging Face](https://huggingface.co/PKUWilliamYang/VToonify/tree/main/models):

<table>
    <tr>
        <th>Backbone</th><th>Model</th><th>Description</th>
    </tr>
    <tr>
        <td rowspan="6">DualStyleGAN</td><td><a href="https://drive.google.com/drive/folders/1DuZfXt6b_xhTAQSN0D8m7N1np0Web0Ky">cartoon</a></td><td>pre-trained VToonify-D models and 317 cartoon style codes</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/drive/folders/12TzTQqwBedsYX3kE_420mdTbWl9lwv4Y">caricature</a></td><td>pre-trained VToonify-D models and 199 caricature style codes</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/drive/folders/1MpEqS26Q1ngTPeex_4MN9qOJxfXKH-k-">arcane</a></td><td>pre-trained VToonify-D models and 100 arcane style codes</td>
    </tr> 
    <tr>
        <td><a href="https://drive.google.com/drive/folders/15mxb7DxTzEBrKtx5aJ_I5WGDjSWBmcUi">comic</a></td><td>pre-trained VToonify-D models and 101 comic style codes</td>
    </tr>   
    <tr>
        <td><a href="https://drive.google.com/drive/folders/1Hld7OeZqYBrg6r35IA_x4sNtt1abHUMU">pixar</a></td><td>pre-trained VToonify-D models and 122 pixar style codes</td>
    </tr> 
    <tr>
        <td><a href="https://drive.google.com/drive/folders/1LQGNMDEHM70nOhm3-xY228YpJNlPnf_s">illustration</a></td><td>pre-trained VToonify-D models and 156 illustration style codes</td>
    </tr>     
    <tr>
        <td rowspan="5">Toonify</td><td><a href="https://drive.google.com/drive/folders/1FFtTVgiDKZ_InnwUJLDuA1wfghZp41nX">cartoon</a></td><td>pre-trained VToonify-T model</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/drive/folders/1ReRxttV-macgV3epC61qg4TQ3FGAhGqG">caricature</a></td><td>pre-trained VToonify-T model</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/drive/folders/1OXU95BOCCT0f6pGbwQ4yQ1EHb2LPd2yb">arcane</td></a><td>pre-trained VToonify-T model</td>
    </tr> 
    <tr>
        <td><a href="https://drive.google.com/drive/folders/1KvawsOXzKgwDM3Z27sagO_KGE_Kc5GZS">comic</td></a><td>pre-trained VToonify-T model</td>
    </tr>   
    <tr>
        <td><a href="https://drive.google.com/drive/folders/19N4ddcTXhXbTEayTbrFc533EktbhOXMz">pixar</td></a><td>pre-trained VToonify-T model</td>
    </tr>   
    <tr>
        <th colspan="2">Supporting model</th><th> </th>
    </tr>
    <tr>
        <td colspan="2"><a href="https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view?usp=sharing">encoder.pt</a></td><td>Pixel2style2pixel encoder to map real faces into Z+ space of StyleGAN</td>
    </tr>  
    <tr>
        <td colspan="2"><a href="https://drive.google.com/file/d/1jY0mTjVB8njDh6e0LP_2UxuRK3MnjoIR/view">faceparsing.pth</a></td><td>BiSeNet for face parsing from <a href="https://github.com/zllrunning/face-parsing.PyTorch">face-parsing.PyTorch</a></td>
    </tr>      
</table>

The downloaded models are suggested to be arranged in [this folder structure](./checkpoint/).

The VToonify-D models are named with suffixes to indicate the settings, where
- `_sXXX`: supports only one fixed style with `XXX` the index of this style.
    - `_s` without `XXX` means the model supports examplar-based style transfer
- `_dXXX`: supports only a fixed style degree of `XXX`.
    - `_d` without `XXX` means the model supports style degrees ranging from 0 to 1
- `_c`: supports color transfer.

### Style Transfer with VToonify-D

**✔ A quick start [HERE](./output#readme)**

Transfer a default cartoon style onto a default face image `./data/077436.jpg`:
```python
python style_transfer.py --scale_image
```
The results are saved in the folder `./output/`, where `077436_input.jpg` is the rescaled input image to fit VToonify (this image can serve as the input without `--scale_image`) and `077436_vtoonify_d.jpg` is the result.

![077436_overview](https://user-images.githubusercontent.com/18130694/189530937-eb468f96-ac02-4f33-8621-03cb93d17e73.jpg)

Specify the content image and the model, control the style with the following options:
- `--content`: path to the target face image or video
- `--style_id`: the index of the style image (find the mapping between index and the style image [here](https://github.com/williamyang1991/DualStyleGAN/tree/main/doc_images)). 
- `--style_degree` (default: 0.5): adjust the degree of style.
- `--color_transfer`(default: False): perform color transfer if loading a VToonify-Dsdc model.
- `--ckpt`: path of the VToonify-D model. By default, a VToonify-Dsd trained on cartoon style is loaded.
- `--exstyle_path`: path of the extrinsic style code. By default, codes in the same directory as `--ckpt` are loaded.
- `--scale_image`: rescale the input image/video to fit VToonify (highly recommend).
- `--padding` (default: 200, 200, 200, 200): left, right, top, bottom paddings to the eye center.

Here is an example of arcane style transfer:
```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --style_id 77 --style_degree 0.5 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt \
       --padding 600 600 600 600     # use large padding to avoid cropping the image
```
![arcane](https://user-images.githubusercontent.com/18130694/189533139-94c3d086-7fe9-49f9-b31f-dbd2a4798e9f.jpg)


Specify `--video` to perform video toonification:
```python
python style_transfer.py --scale_image --content ./data/YOUR_VIDEO.mp4 --video
```
The above style control options (`--style_id`, `--style_degree`, `--color_transfer`) also work for videos.



### Style Transfer with VToonify-T

Specify `--backbone` as ''toonify'' to load and use a VToonify-T model.
```python
python style_transfer.py --content ./data/038648.jpg \
       --scale_image --backbone toonify \
       --ckpt ./checkpoint/vtoonify_t_arcane/vtoonify.pt \
       --padding 600 600 600 600     # use large padding to avoid cropping the image
```
![arcane2](https://user-images.githubusercontent.com/18130694/189540365-d04ffb2a-d72f-4ada-a2a8-89b8ac9ea441.jpg)

In VToonify-T, `--style_id`, `--style_degree`, `--color_transfer`, `--exstyle_path` are not used.

As with VToonify-D, specify `--video` to perform video toonification.

<br/>

## (2) Training VToonify

Download the supporting models to the `./checkpoint/` folder and arrange them in [this folder structure](./checkpoint/):

| Model | Description |
| :--- | :--- |
| [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) | StyleGAN model trained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) |
| [encoder.pt](https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view) | Pixel2style2pixel encoder that embeds FFHQ images into StyleGAN2 Z+ latent code |
| [faceparsing.pth](https://drive.google.com/file/d/1jY0mTjVB8njDh6e0LP_2UxuRK3MnjoIR/view) | BiSeNet for face parsing from [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) |
| [directions.npy](https://drive.google.com/file/d/1HbjmOIOfxqTAVScZOI2m7_tPgMPnc0uM/view) | Editing vectors taken from [LowRankGAN](https://github.com/zhujiapeng/LowRankGAN) for editing face attributes |
| [Toonify](https://drive.google.com/drive/folders/1GZQ6Gs5AzJq9lUL-ldIQexi0JYPKNy8b) \| [DualStyleGAN](https://drive.google.com/drive/folders/1GZQ6Gs5AzJq9lUL-ldIQexi0JYPKNy8b) | pre-trained stylegan-based toonification models|

To customize your own style, you may need to train a new Toonify/DualStyleGAN model following [here](https://github.com/williamyang1991/DualStyleGAN#3-training-dualstylegan).

### Train VToonify-D

Given the supporting models arranged in the [default folder structure](./checkpoint/), we can simply pre-train the encoder and train the whole VToonify-D by running
```python
# for pre-training the encoder
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train_vtoonify_d.py \
       --iter ITERATIONS --stylegan_path DUALSTYLEGAN_PATH --exstyle_path EXSTYLE_CODE_PATH \
       --batch BATCH_SIZE --name SAVE_NAME --pretrain
# for training VToonify-D given the pre-trained encoder
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train_vtoonify_d.py \
       --iter ITERATIONS --stylegan_path DUALSTYLEGAN_PATH --exstyle_path EXSTYLE_CODE_PATH \
       --batch BATCH_SIZE --name SAVE_NAME                  # + ADDITIONAL STYLE CONTROL OPTIONS
```
The models and the intermediate results are saved in `./checkpoint/SAVE_NAME/` and `./log/SAVE_NAME/`, respectively. 

VToonify-D provides the following STYLE CONTROL OPTIONS:
- `--fix_degree`: if specified, model is trained with a fixed style degree (no degree adjustment)
- `--fix_style`: if specified, model is trained with a fixed style image (no examplar-based style transfer)
- `--fix_color`: if specified, model is trained with color preservation (no color transfer)
- `--style_id`: the index of the style image (find the mapping between index and the style image [here](https://github.com/williamyang1991/DualStyleGAN/tree/main/doc_images)). 
- `--style_degree` (default: 0.5): the degree of style.

Here is an example to reproduce the VToonify-Dsd on Cartoon style and the VToonify-D specialized for a mild toonification on the 26th cartoon style:
```python
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 train_vtoonify_d.py \
       --iter 30000 --stylegan_path ./checkpoint/cartoon/generator.pt --exstyle_path ./checkpoint/cartoon/refined_exstyle_code.npy \
       --batch 1 --name vtoonify_d_cartoon --pretrain      
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 train_vtoonify_d.py \
       --iter 2000 --stylegan_path ./checkpoint/cartoon/generator.pt --exstyle_path ./checkpoint/cartoon/refined_exstyle_code.npy \
       --batch 4 --name vtoonify_d_cartoon --fix_color 
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 train_vtoonify_d.py \
       --iter 2000 --stylegan_path ./checkpoint/cartoon/generator.pt --exstyle_path ./checkpoint/cartoon/refined_exstyle_code.npy \
       --batch 4 --name vtoonify_d_cartoon --fix_color --fix_degree --style_degree 0.5 --fix_style --style_id 26
```
Note that the pre-trained encoder is shared by different STYLE CONTROL OPTIONS. VToonify-D only needs to pre-train the encoder once for each DualStyleGAN model.
Eight GPUs are not necessary, one can train the model with a single GPU with larger `--iter`.

**Tips**: [how to find an ideal model] we can first train a versatile model VToonify-Dsd, 
and navigate around different styles and degrees. After finding the ideal setting, we can then train the model specialized in that setting for high-quality stylization.

### Train VToonify-T

The training of VToonify-T is similar to VToonify-D,
```python
# for pre-training the encoder
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train_vtoonify_t.py \
       --iter ITERATIONS --finetunegan_path FINETUNED_MODEL_PATH \
       --batch BATCH_SIZE --name SAVE_NAME --pretrain       # + ADDITIONAL STYLE CONTROL OPTION
# for training VToonify-T given the pre-trained encoder
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train_vtoonify_t.py \
       --iter ITERATIONS --finetunegan_path FINETUNED_MODEL_PATH \
       --batch BATCH_SIZE --name SAVE_NAME                  # + ADDITIONAL STYLE CONTROL OPTION
```
VToonify-T only has one STYLE CONTROL OPTION:
 - `--weight` (default: 1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0): 18 numbers indicate how the 18 layers of the ffhq stylegan model and the finetuned model are blended to obtain the final Toonify model. [Here](https://github.com/williamyang1991/VToonify/blob/edfd68e96eb0c0ab4c31628feef1b667e890a2cd/train_vtoonify_t.py#L30) is the `--weight` we use in the paper for different styles. Please refer to [toonify](https://github.com/justinpinkney/toonify) for the details.

Here is an example to reproduce the VToonify-T model on Arcane style:
```python
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 train_vtoonify_t.py \
       --iter 30000 --finetunegan_path ./checkpoint/arcane/finetune-000600.pt \
       --batch 1 --name vtoonify_t_arcane --pretrain --weight 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1 1 1 1 1 1 1 1 1 1 1
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 train_vtoonify_t.py \
       --iter 2000 --finetunegan_path ./checkpoint/arcane/finetune-000600.pt \
       --batch 4 --name vtoonify_t_arcane --weight 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1 1 1 1 1 1 1 1 1 1 1
```


<br/>

## (3) Results

Our framework is compatible with existing StyleGAN-based image toonification models to extend them to video toonification, and inherits their appealing features for flexible style control. With DualStyleGAN as the backbone, our VToonify is able to transfer the style of various reference images and adjust the style degree in one model.

https://user-images.githubusercontent.com/18130694/189510094-4378caca-e8d9-48e1-9e5d-c8ec038e4bc5.mp4

Here are the color interpolated results of VToonify-D and VToonify-Dc on Arcane, Pixar and Comic styles.

https://user-images.githubusercontent.com/18130694/189510233-b4e3b4f7-5a37-4e0c-9821-a8049ce5f781.mp4

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{yang2022Vtoonify,
  title={VToonify: Controllable High-Resolution Portrait Video Style Transfer},
  author={Yang, Shuai and Jiang, Liming and Liu, Ziwei and Loy, Chen Change},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={6},
  articleno={203},
  pages={1--15},
  year={2022},
  publisher={ACM New York, NY, USA},
  doi={10.1145/3550454.3555437},
}
```

## Acknowledgments

The code is mainly developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN).
