# LiZAD: A Lightweight Zero-Shot Anomaly Detection Framework for Industrial Manufacturing
The official implementation of the paper [LiZAD: A Lightweight Zero-Shot Anomaly Detection Framework for Industrial Manufacturing](https://arxiv.org/abs/2607.01949) accepted at the IEEE International Conference on Omni-Layer Intelligent Systems (COINS 2026).

## Abstract
In modern high-throughput industrial production lines, product configurations and visual characteristics frequently change, making it impractical to collect and annotate data for every new scenario. This dynamic setting makes Zero-Shot Anomaly Detection (ZSAD) particularly suitable, as it enables defect detection without requiring training on target-specific samples. Although recent ZSAD approaches show promising results, they are computationally intensive and thus unsuitable for deployment on resource-constrained devices. We propose LiZAD: a first lightweight framework designed for real-time ZSAD specifically tailored for use on edge devices. The proposed approach pairs the dense and spatially aware visual features of DINOv3, crucial for precise pixel-level localization, with the highly computationally efficient text embeddings of MobileCLIP2, mapping them into a shared latent space via low-memory trainable projection heads. Compared to six state- of-the-art ZSAD models, LiZAD achieves an average memory reduction of 61.5%, a parameter reduction of 74.6%, and a speedup of 3.02× in terms of latency. Despite substantial reductions in computational and memory costs, our approach maintains competitive anomaly detection performance, dropping the average P-AUROC by just 6.4% relative to the best state- of-the-art model across the VisA, BTAD, MPDD, and MVTec- AD datasets. Finally, it is successfully deployed on the NVIDIA Jetson NX and Jetson AGX edge devices, and tested on the real production line of the omitted due to double-blind peer review.

## Installation
Create a new Conda environment and install the required dependencies:

```bash
conda create -n LiZAD python=3.12
conda activate LiZAD
pip install -r requirements.txt
```

## Install MobileCLIP2
To install MobileCLIP2, follow the steps below (see the official [MobileCLIP repository](https://github.com/apple/ml-mobileclip) for more details).

### 1. Install MobileCLIP
Clone the MobileCLIP repository and install it:
```bash
git clone https://github.com/apple/ml-mobileclip.git
cd ml-mobileclip
pip install .
```

### 2. Download the MobileCLIP2-S0 Checkpoint
Download the pretrained MobileCLIP2-S0 checkpoint from Hugging Face:
```bash
hf download apple/MobileCLIP2-S0 --local-dir ./pretrained
```

### 3. Prepare the Dataset
We follow the standard dataset preprocessing pipeline. The framework requires a `meta.json` file in the root directory of each dataset.
To generate the required `meta.json` files, please follow the instructions provided in the [AdaCLIP repository](https://github.com/caoyunkang/AdaCLIP).

### 4. Configure Dataset Paths
Update the `base_dir` variable in `datasets/__init__.py` to point to your dataset root directory.

### 5. Training
Use the following command to train the model:

```bash
bash train.sh
```

### 6. Evaluation
Use the following command to evaluate the trained model:

```bash
bash test.sh
```
### 7. Acknowledgement
We gratefully acknowledge the open-source contributions of the [AdaCLIP](https://github.com/caoyunkang/AdaCLIP), [AA-CLIP](https://github.com/Mwxinnn/AA-CLIP), [MobileCLIP](https://github.com/apple/ml-mobileclip), [DINOv3](https://github.com/facebookresearch/dinov3) authors, whose work made this project possible.

## Authors ##
Uzair Khan<sup>1</sup>, Luigi Capogrosso<sup>2</sup>, Muhammad Aqeel<sup>1</sup>, Francesco Setti<sup>1</sup>, Michele Magno<sup>3,2</sup>, Marco Cristani<sup>1,4 </sup>

<sup>1</sup> *University of Verona*
<sup>2</sup> *Interdisciplinary Transformation University of Austria*
<sup>3</sup> *ETH Zurich*
<sup>4</sup> *Reykjavik University*
