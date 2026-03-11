# Multimodal Machine Learning for Automotive Hood Frame Performance Prediction

[![Paper](https://img.shields.io/badge/arXiv-2508.20358-b31b1b.svg)](https://arxiv.org/abs/2508.20358)
[![Conference](https://img.shields.io/badge/ASME%20IDETC%2FCIE-2025-blue)](https://event.asme.org/IDETC-CIE)

Official repository for the paper:

> **Developing Multi-Modal Machine Learning Model for Predicting Performance of Automotive Hood Frames**
>
> Abhishek Indupally, Satchit Ramnath
> Clemson University, Clemson, SC
>
> *Proceedings of the ASME 2025 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC/CIE 2025), August 17–20, 2025, Anaheim, California. Paper No. DETC2025-168841.*

---

## Overview

Evaluating the structural performance of an automotive hood frame traditionally requires computationally expensive Finite Element Analysis (FEA). This project develops a **Multimodal Machine Learning (MMML)** architecture that learns from multiple data modalities to rapidly predict key performance metrics—replacing or supplementing FEA in the early conceptual design phase.

By combining three complementary data modalities extracted from the same 3D geometry, the MMML model achieves higher accuracy than single-modality (unimodal) approaches and generalizes to hood frame designs that were never seen during training.

### Predicted Performance Metrics

| Metric | Description |
|---|---|
| von Mises Stress (MPa) | Maximum stress in the hood frame under load |
| Directional Deflection (mm) | Maximum displacement under load |
| Geometry Mass (kg) | Total mass of the hood frame |

---

## Architecture

The model integrates **five independent networks** whose outputs are concatenated and passed through fully connected layers for final prediction.

```
Top View Image (128×128×1)   ──► ResNet50 CNN ──────────────────────────────────────────────────────────────────────────────────────────────┐
Side View Image (128×128×1)  ──► ResNet50 CNN ─────────────────────────────────────────────────────────────────────────────────────────────┐ │
Cross-Section @ 25% (600,)   ──► Dense NN (128→256→512→1024→512→128) ──────────────────────────────────────────────────────────────────┐ │ │
Cross-Section @ 75% (600,)   ──► Dense NN (128→256→512→1024→512→128) ─────────────────────────────────────────────────────────────────┐ │ │ │
Rib Depth Parameters (1,)    ──► Dense NN (128→256→512→128) ──────────────────────────────────────────────────────────────────────┐  │ │ │ │
                                                                                                                                   └──┴─┴─┴─┘
                                                                                                                                   Concatenate
                                                                                                                                       │
                                                                          Fully Connected Layers (256→128→64→32→16) + Dropout + L2 Reg
                                                                                                                                       │
                                                                                                              Output: [Stress, Mass, Deflection]
```

### Image Networks
- Two **ResNet50** models (one for top view, one for side view), each trained from random weights on the dataset
- Each outputs a 128-dimensional feature vector via Global Average Pooling → Dense(128, ReLU) → Batch Normalization

### Cross-Section Networks
- Two dense neural networks processing cross-sections at **25%** and **75%** of the hood width
- Each has six hidden layers: 128 → 256 → 512 → 1024 → 512 → 128 nodes (ReLU), with Batch Normalization and a 128-d output

### Parametric (Depth) Network
- One dense neural network processing scalar rib depth parameters
- Four hidden layers: 128 → 256 → 512 → 128 nodes (ReLU), with a 128-d output

### Fusion & Prediction
- All five 128-d feature vectors are concatenated into a single 640-d vector
- Processed through fully connected layers (256 → 128 → 64 → 32 → 16) with L2 regularization and Dropout (60% → 50% → 40% → 30%)
- Final output: 3-node dense layer predicting Stress, Mass, and Deflection

---

## Dataset

The model is trained on the **[CarHoods10K](https://github.com/Fraunhofer-SCAI/CarHoods10k)** dataset, which contains over 10,000 validated 3D mesh automotive hood frame geometries. Each sample includes:

- STL geometry files
- Design parameters (rib depths, pocket dimensions, etc.)
- FEA-derived performance metrics (von Mises stress, directional deflection, mass)

### Data Modalities Used

| Modality | Description | Format |
|---|---|---|
| **Top-view images** | 128×128 grayscale renders of the hood frame from above | `.jpg` |
| **Side-view images** | 128×128 grayscale renders from the side | `.jpg` |
| **Cross-section @ 25%** | (x, z) coordinate profile at 25% of hood width | CSV → padded to 600 values |
| **Cross-section @ 75%** | (x, z) coordinate profile at 75% of hood width | CSV → padded to 600 values |
| **Rib depth** | Scalar depth parameter of primary rib features | CSV |

### Dataset Split

- **Training / Test**: 80% / 20% split from the main dataset
- **Validation**: 20 design points held out entirely from training and testing

---

## Results

### Validation Dataset (CarHoods10K held-out set)

| Output | Average Error |
|---|---|
| vM Stress (MPa) | **5.48%** |
| Geometry Mass (kg) | **9.27%** |
| Directional Deflection (mm) | **16.84%** |

### Generalization to Unseen Hood Frames

Two hood frame designs from [GrabCAD](https://grabcad.com/) (not in CarHoods10K) were used to test generalization:

| Metric | Avg. Error |
|---|---|
| vM Stress (MPa) | 18.52% |
| Geometry Mass (kg) | 32.94% |
| Directional Deflection (mm) | 28.33% |

### Multimodal vs. Unimodal Comparison

| Dataset | Model | Stress Error | Mass Error | Deflection Error |
|---|---|---|---|---|
| CarHoods10K | Unimodal (images only) | 8.2% | 8.1% | 15.2% |
| CarHoods10K | **Multimodal (MMML)** | **3.6%** | **4.7%** | **10.5%** |
| New Hood Frames | Unimodal (images only) | 50.3% | 9.6% | 48.0% |
| New Hood Frames | **Multimodal (MMML)** | **19.4%** | **32.9%** | **28.7%** |

The MMML model consistently outperforms the unimodal baseline, especially on out-of-distribution designs.

---

## Repository Structure

```
mmml-performance/
├── train.py          # Model architecture definition and training script
├── dataset.py        # Dataset preparation: loads images, cross-sections, and depth data; saves .npz files
├── loadnpz.py        # Utility to inspect and explore .npz dataset files
├── save_data.py      # Utility to export .npz arrays to individual image/CSV/npy files
├── stl2image.py      # Utility to render STL geometry files to top/side-view images
├── requirements.txt  # Conda environment specification (tf_gpu)
├── manuscript.tex    # Full LaTeX source for the IDETC/CIE 2025 paper
└── data/             # Directory for dataset files (not included in repo)
```

---

## Setup & Installation

### Requirements

- Python 3.10
- CUDA 11.2 + cuDNN 8.1 (for GPU training)
- TensorFlow 2.10 / Keras 2.10

### Environment Setup

A full Conda environment specification is provided in `requirements.txt`:

```bash
conda env create -f requirements.txt
conda activate tf_gpu
```

---

## Usage

### 1. Prepare the Dataset

Organize your data directories with top-view images, side-view images, performance CSVs, and cross-section CSVs. Then run:

```bash
python dataset.py
```

This will produce `training_dataset.npz` and `validation_dataset.npz` in the configured output directory.

### 2. Train the Model

```bash
python train.py
```

Training configuration (edit directly in `train.py`):

| Parameter | Value |
|---|---|
| Image input size | 128 × 128 × 1 |
| Cross-section input size | 600 |
| Learning rate | 1e-5 (adaptive via `ReduceLROnPlateau`) |
| Batch size | 16 |
| Max epochs | 250 |
| Loss function | Mean Absolute Error (MAE) |
| Optimizer | Adam |

The best model is saved as `best_model.h5`. The final model is saved as `all_inputs_more_layers4.h5`.

### 3. Inspect Dataset Files

To view keys and shapes inside an `.npz` file:

```bash
python loadnpz.py
```

### 4. Export Dataset Arrays

To extract images and CSVs from an `.npz` file:

```bash
python save_data.py
```

---

## Citation

If you use this code or the findings of this paper, please cite:

```bibtex
@inproceedings{indupally2025mmml,
  title     = {Developing Multi-Modal Machine Learning Model for Predicting Performance of Automotive Hood Frames},
  author    = {Indupally, Abhishek and Ramnath, Satchit},
  booktitle = {Proceedings of the ASME 2025 International Design Engineering Technical Conferences
               and Computers and Information in Engineering Conference (IDETC/CIE 2025)},
  number    = {DETC2025-168841},
  year      = {2025},
  address   = {Anaheim, California},
  publisher = {ASME}
}
```

A preprint is available on arXiv: [arXiv:2508.20358](https://arxiv.org/abs/2508.20358)

---

## Acknowledgements

The authors would like to thank **Simon Vaglienti** and **Sandesh Kumawat** for their contributions to this project.

---

## License

See the repository license file for details.
