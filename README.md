# RS-IOD：Scale-decoupled Topology Alignment with Pseudo-label Refinement for Remote Sensing Incremental Object Detection

![STAR-IOD Framework](overall.png)
*Figure 1: Overview of the STAR-IOD framework. It consists of two key components: Subspace-decoupled Topology Distillation (STD) for aligning scale-specific topological structures, and Clustering-driven Pseudo-label Generator (CPG) for adaptive reliable supervision.*
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:

**Subspace-decoupled Topology Distillation for Remote Sensing Object Detection** *Authors: Your Name, Co-author Name, etc.* Published in **CVPR 2024** (or TGRS/ICCV, etc.)

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
