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
**High-Resolution** (>1024) | **Video Toonification** (support unaligned faces) | **Data-Friendly** (no real training data) | **Style Control**
