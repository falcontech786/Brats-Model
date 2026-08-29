# 🧠 BraTS 3D Brain Tumor Segmentation System

AI-powered 3D Brain Tumor Segmentation pipeline built using TensorFlow, 3D U-Net, Hybrid Loss (Dice + Focal Loss), and an interactive Streamlit Web Application.

---

## 🌟 Key Features

- **3D U-Net Architecture**: Deep Learning model designed for 3D multi-modal MRI scans (`FLAIR`, `T1CE`, `T2`).
- **Hybrid Loss Function**: Combines **Dice Loss** and **Categorical Focal Loss** for severe class imbalance handling.
- **Google Colab Pipeline**: High-speed training pipeline optimized for free cloud GPUs (NVIDIA T4 / A100).
- **Interactive Streamlit Web App**: Web GUI for uploading MRI scans (`.nii` / `.nii.gz` / `.npy`) and viewing 3D tumor segmentations interactively.
- **Quantitative Volumetric Analysis**: Calculates voxel counts and volume estimates ($\text{cm}^3$) for Necrotic Core, Peritumoral Edema, and Enhancing Tumor regions.

---

## 📁 Repository Structure

```
├── app.py                          # Streamlit Interactive Web Application
├── colab_brats_3d_unet.py          # Google Colab High-Performance Training Pipeline
├── simple_3d_unet.py               # 3D U-Net Architecture Definition
├── 232_brats2020_get_data_ready.py # NIfTI Preprocessing & Cropping Script
├── 233_custom_datagen.py           # 3D Data Generator with Augmentation
├── 234_train_brats2020_V5.0.py     # Local Training & Metric Evaluation Script
├── requirements.txt                # Python Dependencies
└── .gitignore                      # Git Ignore File
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install required packages:

```bash
git clone <your-repository-url>
cd Brats-Segmenting
pip install -r requirements.txt
```

### 2. Run Training on Google Colab

Open Google Colab, mount Google Drive, and run the self-contained notebook script `colab_brats_3d_unet.py`.

### 3. Launch Web Application

Place your trained model `brats_3d_best.keras` in the project root directory and run:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Tumor Segmentation Classes

- ⬛ **Label 0**: Background / Healthy Brain Tissue
- 🔴 **Label 1**: Necrotic Core & Non-Enhancing Tumor
- 🟢 **Label 2**: Peritumoral Edema
- 🟡 **Label 3**: Enhancing Tumor
