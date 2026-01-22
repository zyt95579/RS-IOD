# RS-IOD：Scale-decoupled Topology Alignment with Pseudo-label Refinement for Remote Sensing Incremental Object Detection

![STAR-IOD Framework](overall.png)
*Figure 1: Overview of the STAR-IOD framework. It consists of two key components: Subspace-decoupled Topology Distillation (STD) for aligning scale-specific topological structures, and Clustering-driven Pseudo-label Generator (CPG) for adaptive reliable supervision.*
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:


##Abstract** *Authors: Your Name, Co-author Name, etc.* Published in **CVPR 2024** (or TGRS/ICCV, etc.)
Remote sensing imagery typically arrives in the form of continuous data streams. Traditional detectors often forget previously learned categories when learning new ones; therefore, research on Remote Sensing Incremental Object Detection (RS-IOD) is of great significance. However, existing methods largely overlook the intra-class scale variations prevalent in remote sensing scenes, which undermines the effectiveness of knowledge transfer and old knowledge preservation. Moreover, RS-IOD also suffers from annotations missing, which cause the model to misclassify old-class instances as background. To address these challenges, we propose a novel framework, STAR-IOD. First, we introduce a Subspace-decoupled Topology Distillation (STD) module to transfer structural knowledge, explicitly aligning inter-class topological relationships and mitigating intra-class representation discrepancies induced by scale shifts. Furthermore, we introduce the Clustering-driven Pseudo-label Generator (CPG), a plug-and-play module that leverages K-Means clustering to dynamically identify class-specific thresholds, thereby guaranteeing an accurate distinction between true positive targets and background noise and alleviating the issue of missing annotations for old classes. We also constructed two Remote Sensing Incremental Object Detection datasets, DIOR-IOD and DOTA-IOD  to facilitate research on RS-IOD. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches by 1.7% and 2.1% mAP on DIOR-IOD and DOTA-IOD, respectively, effectively alleviating catastrophic forgetting while preserving strong detection performance on both base and novel classes. 

![Method Overview](path/to/your/image.png)
*Figure 1: Overview of the Subspace-decoupled Topology Distillation framework.*

## 🔨 Installation

### Requirements
* Linux (Ubuntu 20.04 recommended)
* Python 3.8+
* PyTorch 1.10+
* CUDA 11.3+

### Step-by-step Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/repo-name.git](https://github.com/your-username/repo-name.git)
   cd repo-name
   
## 🚀 Training

### Data Preparation

data/
├── DOTA/
│   ├── train/
│   ├── val/
│   └── annotations/

### Single GPU Training
python tools/train.py configs/std/std_r50_1x_dota.py

## ⚡ Inference / Testing
python tools/test.py configs/std/std_r50_1x_dota.py work_dirs/std_r50_1x_dota/latest.pth --eval mAP
## Acknowledgement
Our code is based on the project MMDetection. Thanks to the work [GCD](https://github.com/Never-wx/GCD).

## 🖊️ Citation
If you find this project useful in your research, please consider citing our paper:
@inproceedings{zhang2024subspace,
  title={Subspace-decoupled Topology Distillation for Remote Sensing Object Detection},
  author={Zhang, San and Li, Si and Wang, Wu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1234--1243},
  year={2024}
}

Acknowledgement
This project is based on MMDetection and GCD. Thanks for their excellent work.
