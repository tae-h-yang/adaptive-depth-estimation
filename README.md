# Enhancing Monocular Metric Depth Estimation through Adaptive Scaling

This repository implements a lightweight adaptive scaling framework to correct scale ambiguity in monocular depth estimation. The method learns to predict an image-specific scaling factor that, when applied to relative depth predictions (e.g., from Depth Anything V2), significantly improves metric depth accuracy — all without extra sensors or modifications to the base depth model.

## 🔧 Setup Instructions

### 1. Clone the repository (with submodules)
```bash
git clone --recursive https://github.com/tae-h-yang/adaptive-depth-estimation.git
cd adaptive-depth-estimation
```

### 2. Set up the environment
Be sure to source the setup script before running anything:
```bash
source setup.sh
```

This installs the required dependencies and sets up environment variables.

### 3. Prepare the dataset
Download the **NYU Depth V2** dataset using [these instructions](https://github.com/wl-zhao/VPD/blob/main/depth/README.md), and place the extracted files into the `datasets/` directory:

### 4. Add model checkpoints
Place the pretrained **Depth Anything V2** checkpoint in the `checkpoints/` directory:

## 🧠 Method Overview

This project introduces a two-stage solution to improve monocular metric depth prediction:

1. **Offline Optimization:** Compute the optimal per-image scale by minimizing the Wasserstein distance between predicted and ground-truth depth distributions.
2. **Online Correction:** Train a lightweight CNN to predict a log-scale correction factor directly from an RGB image.

The predicted scaling factor is applied to the base model's output at inference time, yielding improved metric accuracy in diverse scenes.

## 📂 Directory Structure

```
adaptive_depth/        # Core implementation of the adaptive scaling model
Depth-Anything-V2/     # Submodule for Depth Anything V2
datasets/              # NYU Depth V2 dataset (user-supplied)
checkpoints/           # Pretrained model weights
figures/               # Visualizations and plots
setup.sh               # Setup script for environment
```

## 📜 License

This repository is open-sourced under the MIT License. See [LICENSE](LICENSE) for details.
