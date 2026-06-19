# SatEnhanceAI
AI-powered Satellite Image Enhancement and Super Resolution System
# SatEnhance AI

## Overview

SatEnhance AI is a Deep Learning based Satellite Image Enhancement and Super Resolution system.

The project improves low-quality satellite imagery by:

- Super Resolution
- Noise Reduction
- Blur Removal
- Image Restoration

## Features

- Synthetic degradation pipeline
- Config-driven architecture
- PyTorch Dataset implementation
- Satellite image enhancement
- Future Real-ESRGAN integration

## Project Structure

```text
SatEnhanceAI
│
├── src
│   ├── data
│   │   ├── dataset.py
│   │   └── degradation.py
│   │
│   └── utils
│       └── config.py
│
├── configs
│   └── train_config.yaml
│
├── checkpoints
├── outputs
├── notebooks
└── README.md
```

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Streamlit
- YAML

## Dataset

UC Merced Land Use Dataset

## Status

Phase 2 Completed:
- Data Pipeline
- Image Degradation Pipeline
- Config Management

Upcoming:
- SRCNN Model
- Training Pipeline
- PSNR/SSIM Evaluation
- Streamlit Dashboard
