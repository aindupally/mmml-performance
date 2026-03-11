# mmml-performance

A **Multimodal Machine Learning (MMML)** framework for predicting structural performance metrics of automotive hood frames. The model is trained on three modalities of data — images, cross-sectional geometry, and parametric rib-depth data — extracted from the [CarHoods10K](https://github.com/fsahli/CarHoods10k) dataset, and predicts three FEA-based performance metrics: **von Mises stress**, **directional deflection**, and **geometry mass**.

> **Paper:** Indupally, A., and Ramnath, S., 2025, "Developing Multi-Modal Machine Learning Model for Predicting Performance of Automotive Hood Frames," *Proceedings of the ASME 2025 IDETC/CIE*, DETC2025-168841, August 17–20, 2025, Anaheim, California.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Convert STL Files to Images](#1-convert-stl-files-to-images)
  - [2. Prepare the Dataset](#2-prepare-the-dataset)
  - [3. Train the Model](#3-train-the-model)
  - [4. Inspect NPZ Data Files](#4-inspect-npz-data-files)
  - [5. Export Data from NPZ Files](#5-export-data-from-npz-files)
- [Results](#results)
- [Citation](#citation)

---

## Overview

Evaluating the structural integrity of an automotive hood frame traditionally requires running full Finite Element Analysis (FEA) simulations for every design iteration — a computationally expensive and time-consuming process. This repository implements an MMML model that learns from multiple representations (modalities) of the same geometry to provide rapid performance estimates, significantly reducing reliance on FEA during the conceptual design phase.

**Key contributions:**

- A custom MMML architecture that integrates image, geometric, and parametric data modalities.
- Demonstrated improvement over unimodal (image-only) approaches across all three performance metrics.
- Generalization to unseen hood frame designs sourced from outside the training dataset.

---

## Architecture

The MMML model consists of **five parallel feature-extraction branches** whose outputs are concatenated and passed through a final set of fully connected layers.

```
Top-view Image (128×128×1)   ──► ResNet50 (no top layers) ──► GlobalAvgPool ──► Dense(128) ──┐
Side-view Image (128×128×1)  ──► ResNet50 (no top layers) ──► GlobalAvgPool ──► Dense(128) ──┤
Cross-section @ 25% (600,)   ──► Dense(128,256,512,1024,512,128) ────────────────────────────┤──► Concat ──► FC Layers ──► Output(3)
Cross-section @ 75% (600,)   ──► Dense(128,256,512,1024,512,128) ────────────────────────────┤
Rib Depth Parameter (1,)     ──► Dense(128,256,512,128) ─────────────────────────────────────┘
```

**Final fully connected layers:**  
`Concat(640) → Dense(256) → Dropout(0.6) → Dense(128) → Dropout(0.5) → Dense(64) → Dropout(0.4) → Dense(32) → Dropout(0.3) → Dense(16) → Dense(3)`

**Outputs (3 nodes):**

| Output | Unit |
|---|---|
| von Mises Stress | MPa |
| Geometry Mass | kg |
| Directional Deflection | mm |

**Training configuration:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 1×10⁻⁵ |
| Loss function | Mean Absolute Error (MAE) |
| Batch size | 16 |
| Max epochs | 250 |
| LR schedule | ReduceLROnPlateau (factor=0.9, patience=2) |
| Regularization | L2 (λ=0.01) + Dropout |

---

## Dataset

The model is trained on the **CarHoods10K** dataset — a publicly available collection of over 10,000 validated 3D automotive hood frame geometries in STL format, each annotated with design parameters and FEA performance results (stress, deflection, mass).

Three data modalities are extracted from each design:

| Modality | Description | Format |
|---|---|---|
| **Top-view image** | 128×128 grayscale projection along the normal direction | `.jpg` / NumPy array |
| **Side-view image** | 128×128 grayscale projection along the lateral direction | `.jpg` / NumPy array |
| **Cross-section @ 25%** | (x, z) coordinate points at 25% of frame width, padded to length 600 | CSV / NumPy array |
| **Cross-section @ 75%** | (x, z) coordinate points at 75% of frame width, padded to length 600 | CSV / NumPy array |
| **Rib depth** | Scalar depth parameter of primary rib features | CSV / NumPy scalar |

The cross-sections use a **10/10/80 sampling strategy**: 10% of points from the start, 10% from the end, and 80% uniformly sampled from the middle, padded or truncated to 600 values.

**Dataset splits:**

- 20 design points held out as a **validation** set (never seen during training or testing).
- The remaining data is split **80/20** for training and testing.

The processed dataset is stored as two `.npz` files:

```
data/
  training_dataset.npz    # top_view_images, side_view_images, depth_data,
                           # cross_section_25, cross_section_75, performance_data
  validation_dataset.npz  # same keys, 20 held-out design points
```

---

## Repository Structure

```
mmml-performance/
├── data/                    # NPZ dataset files (training & validation)
├── output/                  # Output directory for generated images or results
├── stl2image.py             # Convert STL geometry files to 2D top/side-view images
├── dataset.py               # Build training_dataset.npz and validation_dataset.npz
│                            #   from raw images and CSV files
├── train.py                 # Define and train the MMML model
├── loadnpz.py               # Inspect contents and shapes of an NPZ file
├── save_data.py             # Export arrays from an NPZ file to images/CSVs/npy
├── requirements.txt         # Conda environment specification (tf_gpu)
└── manuscript.tex           # LaTeX source of the ASME 2025 conference paper
```

---

## Setup and Installation

The project requires **Python 3.10**, **TensorFlow 2.10**, and **CUDA 11.2 / cuDNN 8.1** for GPU-accelerated training.

**1. Create the conda environment from the provided specification:**

```bash
conda env create -f requirements.txt
conda activate tf_gpu
```

> The `requirements.txt` is a conda environment YAML (`name: tf_gpu`). It pins all dependencies including CUDA, cuDNN, TensorFlow 2.10, Keras 2.10, scikit-learn, matplotlib, and NumPy.

**2. Verify GPU availability (optional):**

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Usage

### 1. Convert STL Files to Images

Use `stl2image.py` to generate top-view and side-view grayscale PNG images from STL geometry files.

```bash
python stl2image.py
```

By default this reads `.stl` files from `data/ch10k/` and writes `<name>_top.png` / `<name>_side.png` pairs to `./output/`. Edit the `stl_directory` variable at the bottom of the script to point to your STL files.

---

### 2. Prepare the Dataset

Use `dataset.py` to build the `.npz` training and validation files from:

- Top-view and side-view image folders (`.jpg` files)
- Performance / depth CSV files
- Cross-section CSV folders (25% and 75% locations)

Edit the path variables near the bottom of `dataset.py` to point to your data directories, then run:

```bash
python dataset.py
```

This creates `training_dataset.npz` and `validation_dataset.npz` in the specified output directory.

---

### 3. Train the Model

```bash
python train.py
```

Update the `npz_file` path at the bottom of `train.py` to point to your `training_dataset.npz`. The script will:

1. Load the dataset from the `.npz` file.
2. Split it 80/20 into train and test sets.
3. Build and compile the MMML model.
4. Train for up to 250 epochs with adaptive learning rate.
5. Plot training/validation loss curves.
6. Save the final model to `all_inputs_more_layers4.h5`.

---

### 4. Inspect NPZ Data Files

```bash
python loadnpz.py
```

By default inspects `data/validation_dataset.npz`. Edit the `npz_file` path in the script to point to a different file. Prints the shape and dtype of every stored array.

---

### 5. Export Data from NPZ Files

`save_data.py` extracts each array from an `.npz` file and saves it in a human-readable format:

- Arrays with `image` in the key name → `.png` files
- Arrays with `cross`, `section`, `depth`, or `performance` in the key name → `.csv` files
- All other arrays → `.npy` files

```bash
python save_data.py
```

Edit `npz_file` and `output_directory` at the bottom of the script as needed.

---

## Results

### Validation on CarHoods10K (held-out set, 20 design points)

| Output Parameter | Average Error |
|---|---|
| von Mises Stress | 5.48 % |
| Geometry Mass | 9.27 % |
| Directional Deflection | 16.84 % |

### Multimodal vs. Unimodal (image-only multi-view CNN)

| Dataset | Model | Stress Error | Mass Error | Deflection Error |
|---|---|---|---|---|
| CarHoods10K | Unimodal | 8.2 % | 8.1 % | 15.2 % |
| CarHoods10K | **Multimodal** | **3.6 %** | **4.7 %** | **10.5 %** |
| New (unseen) frames | Unimodal | 50.3 % | 9.6 % | 48.0 % |
| New (unseen) frames | **Multimodal** | **19.4 %** | 32.9 % | **28.7 %** |

The MMML model consistently outperforms the unimodal baseline, most significantly on out-of-distribution (unseen) hood frame designs, demonstrating strong generalization.

### Prediction on Two Unseen Hood Frames (from GrabCAD)

| Output | Frame 1 Actual | Frame 1 Predicted | Frame 2 Actual | Frame 2 Predicted | Avg. Error |
|---|---|---|---|---|---|
| vM Stress (MPa) | 207.5 | 156.88 | 152.49 | 133.19 | 18.52 % |
| Geometry Mass (kg) | 9.68 | 13.24 | 10.53 | 13.60 | 32.94 % |
| Dir. Deflection (mm) | 10.03 | 7.78 | 11.74 | 7.78 | 28.33 % |

---

## Citation

If you use this code or the approach described here, please cite the associated paper:

```bibtex
@inproceedings{indupally2025mmml,
  author    = {Indupally, Abhishek and Ramnath, Satchit},
  title     = {Developing Multi-Modal Machine Learning Model for Predicting Performance of Automotive Hood Frames},
  booktitle = {Proceedings of the ASME 2025 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC/CIE2025)},
  number    = {DETC2025-168841},
  year      = {2025},
  month     = {August},
  address   = {Anaheim, California}
}
```
