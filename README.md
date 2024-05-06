# DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Indistinct-Boundary Object Segmentation

### [Project page](https://github.com/IMOP-lab/DEFN-pytorch) | [Paper](https://arxiv.org/abs/2311.00483) | [Our laboratory home page](https://github.com/IMOP-lab) 

DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Indistinct-Boundary Object Segmentation.<be>

Xingru Huang, Yihao Guo, Jian Huang, Zhi Li, Tianyun Zhang, Kunyan Cai, Gaopeng Huang, Wenhao Chen, Zhaoyang Xu, Liangqiong Qu, Ji Hu, Tinyu Wang, Shaowei Jiang, Chenggang Yan, Yaoqi Sun, Xin Ye, Yaqi Wang

Hangzhou Dianzi University

<div align=center>
  <img src="https://github.com/IMOP-lab/DEFN-pytorch/blob/main/images/System_structure.png">
</div>
<p align=center>
  Figure 1: The system structure.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/DEFN-pytorch/blob/main/images/Network_structure.png">
</div>
<p align=center>
  Figure 2: The network structure of DEFN. 
</p>

We proposed DEFN, a 3D OCT segmentation network for eye diseases with unobvious characteristics that are easily interfered with, such as macular holes and macular edema.

This repository contains the official Pytorch implementation for DEFN and DWC Loss, as well as the pre-trained model for DEFN.

## Methods
### FuGH Module

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/FuGH.png"width=70% height=70%>
</div>
<p align=center>
  Figure 3: The FuGH module.
</p>

The Fourier Group Harmonics (FuGH) module enhances noise reduction in OCT sequences by employing FFT for feature extraction in the frequency domain, enabling targeted noise filtration and efficient processing of periodic patterns.

### S3DSA Module

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/S3DSA.png"width=70% height=70%>
</div>
<p align=center>
  Figure 4: The S3DSA module.
</p>

The Simplified 3D Spatial Attention (S3DSA) module improves the segmentation of macular holes and edema in fundus OCT sequences by an optimized spatial attention mechanism. It refines focus on crucial regions, enhancing segmentation quality and computational efficiency.

### HSE Module

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/HSE.png"width=70% height=70%>
</div>
<p align=center>
  Figure 5: The HSE module.
</p>

The Harmonic Squeeze-and-Excitation Module (HSE) combines FuGH and Squeeze-and-Excitation (SE) blocks to enhance the segmentation performance of macular holes and macular edema, by extending the model's view field and recalibrating feature weights.

## Experiment
### Baselines

We have provided the GitHub links to the PyTorch implementation code for all networks compared in the experiments herein.

[3D UX-Net](https://github.com/MASILab/3DUX-Net), [nnFormer](https://github.com/282857341/nnFormer), [3D U-Net](https://github.com/wolny/pytorch-3dunet), [SegResNet](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/segresnet.py), [SwinUNETR](https://github.com/LeonidAlekseev/Swin-UNETR), [TransBTS](https://github.com/Rubics-Xuan/TransBTS), [UNETR](https://github.com/tamasino52/UNETR), [DeepResUNet](https://github.com/xiaorui531/deepresunet_brats), [ResUNet](https://github.com/rishikksh20/ResUnet), [HighRes3DNet](https://github.com/fepegar/highresnet), [MultiResUNet](https://github.com/nibtehaz/MultiResUNet), [SegCaps](https://github.com/lalonderodney/SegCaps), [V-Net](https://github.com/mattmacy/vnet.pytorch)

### Training Results

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/train_Isolated.png"width=100% height=100%>
</div>
<p align=center>
  Figure 6: The Training Results using Isolated Strategy.
</p>

Segmentation results employing the isolated macular hole injection method, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes:  All (Average across all classes), MH (Macular Hole), ME (Macular Edema) and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/train_Comprehensive.png"width=100% height=100%>
</div>
<p align=center>
  Figure 7: The Training Results using the Comprehensive Strategy.
</p>

Segmentation results employing the comprehensive macular hole injection method, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes: All (Average across all classes), MH (Macular Hole), ME (Macular Edema), and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.

### Fine-tuning Results

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/finetune_Isolated.png"width=100% height=100%>
</div>
<p align=center>
  Figure 8: The Fine-tuning Results using the Isolated Strategy.
</p>

Segmentation results of fine-tuning after isolated macular hole injection training, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes: All (Average across all classes), MH (Macular Hole), ME (Macular Edema), and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/finetuen_Comprehensive.png"width=100% height=100%>
</div>
<p align=center>
  Figure 9: The Fine-tuning Results using the Comprehensive Strategy.
</p>

Segmentation results of fine-tuning after comprehensive macular hole injection training, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes: All (Average across all classes), MH (Macular Hole), ME (Macular Edema), and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.

### Ablation Results

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/ablation.png"width=100% height=100%>
</div>
<p align=center>
  Figure 10: The Ablation Results.
</p>

Ablation study on the backbone and the proposed methods, including DEFN (DEFN backbone), HSE (Harmonic Squeeze-and-Excitation module), FuGH (Fourier Group Harmonics module), IMHI (isolated macular hole injection), CMHI (comprehensive macular hole injection) and DWC (Dynamic Weight Compose Loss). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue.

## 3D Reconstruction Results

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/Retinal_3D_%20reconstruction_Results.png"width=100% height=100%>
</div>
<p align=center>
  Figure 11: The 3D reconstruction results.
</p>

Five cases are presented with their original images and reconstruction outcomes. The first row exhibits the original images for each case. Rows two to five show the reconstructions based on four different rendering styles, while the sixth row provides a top view of the reconstruction results. Within the reconstructions, yellow regions indicate macular holes and blue regions signify macular edema.
