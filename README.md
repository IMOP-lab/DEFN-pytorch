# DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Indistinct-Boundary Object Segmentation

## [Project page](https://github.com/IMOP-lab/DEFN-pytorch) | [Paper](https://arxiv.org/abs/2311.00483) | [Our laboratory home page](https://github.com/IMOP-lab) 

**DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Indistinct-Boundary Object Segmentation.**

**Xiaohua Jiang, Yihao Guo, Jian Huang, Yuting Wu, Meiyi Luo, Zhaoyang Xu, Qianni Zhang, Xingru Huang, Hong He, Shaowei Jiang, Jing Ye, Mang Xiao**

**Sir Run Run Shaw Hospital, Hangzhou Dianzi University**

### System structure
<div align=center>
  <img src="https://github.com/IMOP-lab/DEFN-pytorch/blob/main/images/System_structure.png">
</div>
<p align=center>
  Fig. 1: The system structure of our proposed 3D segmentation architecture.
</p>

### Network structure
<div align=center>
  <img src="https://github.com/IMOP-lab/DEFN-pytorch/blob/main/images/Network_structure.png">
</div>
<p align=center>
  Fig. 2: The network structure of our proposed DEFN. 
</p>

**We proposed DEFN, a 3D OCT segmentation network for indistinct-boundary object segmentation with unobvious characteristics that are easily interfered with, such as macular holes and macular edema.**

**This repository contains the official Pytorch implementation for DEFN and DWC Loss, as well as the pre-trained model for DEFN.**

## Main Methods
### FuGH Module
<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/FuGH.png"width=60% height=60%>
</div>
<p align=center>
  Fig. 3: The detailed structure of the FuGH module.
</p>

**The Fourier Group Harmonics (FuGH) module enhances noise reduction in medical image sequences by employing FFT for feature extraction in the frequency domain, enabling targeted noise filtration and efficient processing of periodic patterns.**

### S3DSA Module

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/S3DSA.png"width=60% height=60%>
</div>
<p align=center>
  Fig. 4: The detailed structure of the S3DSA module.
</p>

**The Simplified 3D Spatial Attention (S3DSA) module improves the segmentation of macular holes and edema in fundus OCT sequences by an optimized spatial attention mechanism. It refines focus on crucial regions, enhancing segmentation quality and computational efficiency.**

### HSE Module

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/HSE.png"width=60% height=60%>
</div>
<p align=center>
  Fig. 5: The detailed structure of the HSE module.
</p>

**The Harmonic Squeeze-and-Excitation Module (HSE) combines FuGH and Squeeze-and-Excitation (SE) blocks to enhance the segmentation performance of macular holes and macular edema, by extending the model's view field and recalibrating feature weights.**

## Experiment
### Baselines

**We have provided the GitHub links to the PyTorch implementation code for all networks compared to the experiments herein.**

[3D UX-Net](https://github.com/MASILab/3DUX-Net), [nnFormer](https://github.com/282857341/nnFormer), [3D U-Net](https://github.com/wolny/pytorch-3dunet), [SegResNet](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/segresnet.py), [SwinUNETR](https://github.com/LeonidAlekseev/Swin-UNETR), [TransBTS](https://github.com/Rubics-Xuan/TransBTS), [UNETR](https://github.com/tamasino52/UNETR), [DeepResUNet](https://github.com/xiaorui531/deepresunet_brats), [ResUNet](https://github.com/rishikksh20/ResUnet), [HighRes3DNet](https://github.com/fepegar/highresnet), [MultiResUNet](https://github.com/nibtehaz/MultiResUNet), [SegCaps](https://github.com/lalonderodney/SegCaps), [V-Net](https://github.com/mattmacy/vnet.pytorch)

### Training Results

<div>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/train_Isolated.png"width=100% height=100%>
</div>
<p align=center>
  Fig. 6: The Training Results using Isolated Strategy.
</p>

**Segmentation results employing the isolated macular hole injection method, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes:  All (Average across all classes), MH (Macular Hole), ME (Macular Edema), and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.**

<div>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/train_Comprehensive.png"width=100% height=100%>
</div>
<p align=center>
  Fig. 7: The Training Results using the Comprehensive Strategy.
</p>

**Segmentation results employing the comprehensive macular hole injection method, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes: All (Average across all classes), MH (Macular Hole), ME (Macular Edema), and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.**

### Fine-tuning Results

<div>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/finetune_Isolated.png"width=100% height=100%>
</div>
<p align=center>
  Fig. 8: The Fine-tuning Results using the Isolated Strategy.
</p>

**Segmentation results of fine-tuning after isolated macular hole injection training, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes: All (Average across all classes), MH (Macular Hole), ME (Macular Edema), and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.**

<div>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/Tables/finetuen_Comprehensive.png"width=100% height=100%>
</div>
<p align=center>
  Fig. 9: The Fine-tuning Results using the Comprehensive Strategy.
</p>

**Segmentation results of fine-tuning after comprehensive macular hole injection training, comparing the proposed DEFN, DEFN+DWC Loss, and prior segmentation models. The evaluation spans four classes: All (Average across all classes), MH (Macular Hole), ME (Macular Edema), and RA (Retina). The best values for each metric are highlighted in red, while the second-best values are highlighted in blue, and the values of our model are bolded.**

## 3D Reconstruction Results

<div align=center>
  <img src="https://github.com/IIPL-HangzhouDianziUniversity/DEFN-Pytorch/blob/main/images/Retinal_3D_%20reconstruction_Results.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 10: The 3D reconstruction results.
</p>

**Five cases are presented with their original images and reconstruction outcomes. The first row exhibits the original images for each case. Rows two to five show the reconstructions based on four different rendering styles, while the sixth row provides a top view of the reconstruction results. Within the reconstructions, yellow regions indicate macular holes and blue regions signify macular edema.**

## License
This project is licensed under the [MIT license](https://github.com/IMOP-lab/DEFN-Pytorch/blob/main/LICENSE).



