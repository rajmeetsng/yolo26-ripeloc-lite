# YOLO26-RipeLoc Lite

**A Lightweight Architecture for Tomato Ripeness Detection and Picking Point Localization in Greenhouse Robotic Harvesting**

[![Paper](https://img.shields.io/badge/Paper-Computers%20%26%20Electronics%20in%20Agriculture-blue)](https://doi.org/XXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org)

## Overview

YOLO26-RipeLoc Lite is a lightweight single-stage detection architecture for simultaneous:
- **Tomato detection** in greenhouse environments
- **Binary ripeness classification** (ripe vs. unripe)
- **2D image-plane center-point localization** for robotic harvesting

The model introduces three task-specific modifications to the YOLO26 framework:
1. **LFPN** — Lightweight Feature Pyramid Network with depthwise separable C3k2 blocks
2. **RAAM** — Ripeness-Aware Attention Module with learnable chrominance bias
3. **CDH + CPL** — Compact Detection Head with Center-Point Localization module

![Architecture](docs/architecture.png)

## Key Results

| Metric | Value |
|--------|-------|
| mAP@50 | 92.9% (ranking 4th among 8 models) |
| Precision | **95.2%** (highest among all models) |
| Recall | 92.6% |
| Parameters | **2.38M** (smallest footprint) |
| GFLOPs | 6.4 |
| CPL RMSE | 4.86 px (3.97 mm at 500 mm) |
| CPL <5px accuracy | 79.4% |

## Installation

```bash
git clone https://github.com/[your-username]/yolo26-ripeloc-lite.git
cd yolo26-ripeloc-lite
pip install -r requirements.txt
```

## Quick Start

### Training (3-Phase Progressive Unfreezing)

```bash
# Phase 1: Frozen backbone (50 epochs)
python scripts/train.py --phase 1 --data configs/tomato_dataset.yaml --cfg configs/yolo26n-ripeloc.yaml

# Phase 2: Partial unfreeze (80 epochs)
python scripts/train.py --phase 2 --data configs/tomato_dataset.yaml --resume runs/phase1/weights/last.pt

# Phase 3: Full unfreeze (120 epochs)
python scripts/train.py --phase 3 --data configs/tomato_dataset.yaml --resume runs/phase2/weights/last.pt
```

### Or run all 3 phases automatically:

```bash
python scripts/train_3phase.py --data configs/tomato_dataset.yaml --cfg configs/yolo26n-ripeloc.yaml --seed 0
```

### Inference with CPL

```bash
python scripts/inference.py --weights runs/phase3/weights/best.pt --source path/to/images/ --save-cpl
```

### Evaluation

```bash
python evaluation/evaluate_full.py --weights runs/phase3/weights/best.pt --data configs/tomato_dataset.yaml
```

### BN Channel Pruning

```bash
# Soft pruning + fine-tuning
python scripts/prune_bn.py --weights runs/phase3/weights/best.pt --ratio 0.3 --finetune-epochs 30

# Structural pruning + ONNX export
python scripts/prune_structural.py --weights runs/pruned/soft_pruned.pt --export onnx
```

## Repository Structure

```
yolo26-ripeloc-lite/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── yolo26n-ripeloc.yaml          # Model architecture YAML
│   ├── tomato_dataset.yaml           # Dataset configuration
│   └── training_hyperparams.yaml     # All hyperparameters
├── scripts/
│   ├── train.py                      # Single-phase training
│   ├── train_3phase.py               # 3-phase progressive unfreezing
│   ├── inference.py                  # Detection + CPL inference
│   ├── prune_bn.py                   # BN soft pruning + fine-tune
│   ├── prune_structural.py           # Structural pruning + export
│   └── cpl_module.py                 # Center-Point Localization module
├── evaluation/
│   ├── evaluate_full.py              # Full evaluation pipeline
│   ├── evaluate_cpl.py               # CPL RMSE/MAE evaluation
│   ├── bootstrap_ci.py               # Bootstrap confidence intervals
│   ├── stratified_errors.py          # Error analysis by condition
│   └── inter_annotator.py            # Inter-annotator variability
├── models/
│   ├── raam.py                       # RAAM module implementation
│   ├── lfpn.py                       # LFPN module implementation
│   └── cdh.py                        # CDH + CPL implementation
├── data/
│   ├── splits/
│   │   ├── train.txt                 # Training image paths (1050)
│   │   ├── val.txt                   # Validation image paths (225)
│   │   ├── test.txt                  # Test image paths (225)
│   │   └── cpl_subset.txt            # CPL evaluation subset (48)
│   └── annotations_example/
│       ├── ripe_example.txt          # Annotation format example
│       └── unripe_example.txt        # Annotation format example
└── docs/
    ├── architecture.png
    └── ANNOTATION_GUIDE.md
```

## Dataset

The 1,500-image greenhouse tomato dataset (6,227 annotations) is available upon request with a Data Use Agreement (DUA). Contact: irfan.hussain@ku.ac.ae

| Split | Images | Ripe | Unripe | Total |
|-------|--------|------|--------|-------|
| Train | 1050 | 2487 | 1863 | 4350 |
| Val | 225 | 538 | 396 | 934 |
| Test | 225 | 541 | 402 | 943 |
| **Total** | **1500** | **3566** | **2661** | **6227** |

## Random Seeds

| Experiment | Seed |
|-----------|------|
| Primary results (all tables) | 0 |
| Multi-seed run 2 | 42 |
| Multi-seed run 3 | 123 |

## Citation

```bibtex
@article{singh2026yolo26ripeloc,
  title={YOLO26-RipeLoc Lite: A Lightweight Architecture for Tomato Ripeness
         Detection and Picking Point Localization in Greenhouse Robotic Harvesting},
  author={Singh, Rajmeet and Kaur, Manveen and Alirezaee, Shahpour and Hussain, Irfan},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

This work was supported by KUCARS Theme 4 (RC1-2018-KUCARS) and Silal Innovation Oasis (grant 8475000024).
