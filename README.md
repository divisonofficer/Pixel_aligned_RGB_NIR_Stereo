# Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision

> **CVPR 2025**  
> **Jinnyeong Kim**, **Seung-Hwan Baek**  
> POSTECH  
> [[arXiv]](https://arxiv.org/abs/2411.18025) • [[Code]](https://github.com/your-repo-url) • [[Video]](https://your-video-link.com) • [[Dataset on HuggingFace]](https://huggingface.co/datasets/your-dataset-url)

---

## Overview

This repository provides the code and dataset accompanying our CVPR 2025 paper:

**"Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision"**

We propose a novel robotic vision system equipped with **two pixel-aligned RGB-NIR stereo cameras** and a **LiDAR sensor** mounted on a mobile robot. Our system captures **RGB-NIR stereo video sequences** and **temporally synchronized LiDAR point clouds**, offering a high-quality, aligned multi-spectral dataset under diverse lighting conditions.

![System Overview](https://divisonofficer.github.io/project_page_cvpr2025_5308/fig_imaging_1.png)

---

## ✨ Highlights

- **Pixel-aligned RGB-NIR stereo imaging** for robust vision under challenging lighting.
- **Continuous video sequences** recorded using a mobile robot.
- **Sparse LiDAR point clouds** temporally synchronized with stereo imagery.
- Two proposed methods to utilize RGB-NIR pairs:
  - RGB-NIR **Image Fusion** (pretrained model-compatible)
  - RGB-NIR **Feature Fusion** (for fine-tuned stereo depth estimation)

---

## 📦 Dataset

We release a large-scale dataset for training and evaluating robot vision models in realistic environments.

### 📹 Data Statistics

| | #Videos | #Frames |
|---|--------|---------|
| Training | 80 | 90,000 |
| Testing  | 40 | 7,000 |

### 📁 Per Frame Data Includes:

- Pixel-aligned **RGB-NIR stereo images**
- **Sparse LiDAR** point cloud (in camera coordinates)
- **Sensor timestamps** (synchronized)

### 🌗 Lighting Scenarios
<img width="920" alt="image" src="https://github.com/user-attachments/assets/a07bea4e-5674-4277-a585-f556ce9d4825" />


➡️ **[Download Dataset on HuggingFace](https://huggingface.co/datasets/your-dataset-url)**
Each .tar.gz file follows below structure
```
frame1
--rgb
-----left_distorted.png (or left.png)
-----right_distorted.png (or right.png)
--nir
-----left_distorted.png (or left.png)
-----right_distorted.png (or right.png)
storage.hdf5
```
The frame ids are named after their creation date.
**_distorted.png** image need to be undistorted. **left.png** and **right.png** are undistorted version.

**storage.hdf5** is H5 database. it contains **frame** group with children of each frame ids.

---

## 📷 Imaging System

Our robotic platform integrates:

- **Two RGB-NIR stereo cameras** (pixel-aligned RGB and NIR sensors)
- **LiDAR sensor**
- **Omnidirectional mobile base** (360° movement)
- **High-capacity battery** (up to 6 hours)
- **NIR LED bar light source** for consistent active illumination

![Robot Platform](https://divisonofficer.github.io/project_page_cvpr2025_5308/fig_imaging_1.png)

---

## 🔧 Methods

### RGB-NIR synthetic data augmentation

![image](https://github.com/user-attachments/assets/00805f64-44cf-4ac4-927c-a01ace160f39)

See **visualize/synth_aug_render.ipynb** for method of synthetic data augmentation to build RGB-NIR training dataset. 


### RGB-NIR Image Fusion
![image](https://github.com/user-attachments/assets/0d524c12-8419-48d0-8c3a-0b8a9bc29d1b)

We introduce an RGB-NIR **image-level fusion technique** for 3-channel vision tasks. This approach allows existing **RGB-pretrained models** to benefit from NIR information **without additional fine-tuning**.

Applicable to:
- Stereo Depth Estimation
- Semantic Segmentation
- Object Detection

See **net/image_fusion.py** for pytorch implementation.

### RGB-NIR Feature Fusion (Stereo Depth)

We extend RAFT-Stereo with a novel **feature-level fusion strategy**, alternating between fused and NIR **correlation volumes** during iterative disparity estimation using GRUs.

![image](https://github.com/user-attachments/assets/ef954e60-02d4-4a6c-b126-150ee2edeffc)

See **net/feature_fusion.py** of implementation with RAFT-Stereo as baseline
Our setup reflects the **RGB with active illumination** scenario:  
- NIR provides robust depth cues  
- RGB complements NIR with texture under normal lighting

---

## 📊 Experimental Results

Our experiments demonstrate that pixel-aligned RGB-NIR inputs:
- Improve stereo depth accuracy under low-light and high-contrast conditions
- Enable pretrained RGB models to generalize better
- Enhance robustness across lighting domains




---





## 📄 Citation

If you use this dataset or code, please cite our work:

```bibtex
@article{kim2025pixelnir,
  author    = {Jinnyeong Kim and Seung-Hwan Baek},
  title     = {Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision},
  conference = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  doi       = {10.48550/arXiv.2411.18025},
  url       = {https://arxiv.org/abs/2411.18025},
}
