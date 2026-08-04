# Polyp Segmentation with Fixed-Range Seed-Guided Local Anchoring

This repository contains the training and evaluation code for endoscopic polyp segmentation on the Kvasir-SEG dataset.

The project uses a U-Net with an ImageNet-pretrained ResNet-34 encoder. A training-free fixed-range seed-guided local anchoring strategy is evaluated for false-positive suppression.

> The current code may still use `PRAP` or `prap` as directory and method names for compatibility.

## Environment

The experiments were conducted using:

- Python 3.10.19
- PyTorch 2.10.0+cu128
- NVIDIA GeForce RTX 5060 Laptop GPU
- GPU memory: 8 GB
- NVIDIA driver: 595.95

Main dependencies:

- Albumentations 2.0.8
- segmentation-models-pytorch 0.5.0
- OpenCV 4.13.0
- NumPy 2.2.6
- Pillow 12.0.0
- SciPy 1.15.3
- pandas 2.3.3
- Matplotlib 3.10.8
- tqdm 4.67.2
- MedPy 0.5.2

## Installation

Create the environment:

```bash
conda create -n polypseg python=3.10.19
conda activate polypseg
```

Install PyTorch with CUDA 12.8:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download the Kvasir-SEG dataset and organize it as:

```text
Kvasir-SEG/
├── images/
└── masks/
```

The image and mask filenames must match.

Set the dataset path in:

```text
train/train_common.py
```

Example:

```python
DATA_ROOT = r"D:\datasets\Kvasir-SEG"
```

Endoscopic images are loaded as RGB images. JPEG-format masks are loaded as grayscale images and binarized using a threshold of 128.

The fixed data split contains:

- 700 training images
- 150 validation images
- 150 test images

The split files are stored in:

```text
data_split/splits/
```

## Training

Run the baseline training from the `train` directory:

```bash
cd train
python train_unet.py
```

The best checkpoint is saved to:

```text
train/runs/kvasir_unet_baseline/best.pth
```

Main training settings:

| Parameter | Value |
|---|---:|
| Input size | 352 × 352 |
| Batch size | 4 |
| Maximum epochs | 40 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | StepLR |
| Step size | 10 |
| Gamma | 0.1 |
| Early-stopping patience | 8 |
| Random seed | 42 |

The training loss is:

```text
0.5 × BCEWithLogitsLoss + 0.5 × Soft Dice Loss
```

## Evaluation

Evaluate the baseline model:

```bash
cd train
python evaluate.py --split val --models baseline
python evaluate.py --split test --models baseline
```

Evaluate the fixed-range seed-guided local anchoring strategy from the repository root:

```bash
python PRAP/evaluate_postprocess.py ^
  --split test ^
  --methods prap ^
  --candidate-threshold 0.50 ^
  --seed-threshold 0.95 ^
  --kernel-size 3 ^
  --dilation-iterations 5 ^
  --min-area 320 ^
  --output-dir PRAP/results_test/local_anchoring
```

Formal post-processing parameters:

| Parameter | Value |
|---|---:|
| Candidate threshold | 0.50 |
| Seed threshold | 0.95 |
| Kernel | 3 × 3 ellipse |
| Dilation iterations | 5 |
| Minimum area | 320 pixels |

## Main Test Results

Results on the 150-image test set:

| Method | Dice | Precision | Recall | Total FP | FP reduction |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9141 | 0.9259 | 0.9280 | 213358 | 0.00% |
| Local anchoring | 0.9145 | 0.9389 | 0.9146 | 149035 | 30.15% |

The method reduces false-positive pixels and improves Precision, while Recall decreases.

## Repository Structure

```text
.
├── data_split/
├── train/
├── PRAP/
├── requirements.txt
└── README.md
```