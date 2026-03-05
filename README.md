# Edge-Guided Geometry Encoding Volume for Stereo Matching

This repository contains the official PyTorch implementation for our submission to **ECCV**. 
The code is currently anonymized for double-blind review. 

This work introduces an edge-guided framework built upon Iterative Geometry Encoding Volumes to improve stereo matching, particularly focusing on handling depth-discontinuity and object boundaries.

## 📊 Datasets

To train and evaluate the model, you will need to download the following datasets:
* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (FlyingThings3D, Driving & Monkaa)
* [KITTI 2012 & 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)

*Data Structure:* By default, the codebase assumes datasets are located under `./data/`. Please update the paths in `core/stereo_datasets.py` if your data is stored elsewhere.

## 🏆 Main Results

*Note: Below are the summarized quantitative results of our method. For detailed comparisons and ablation studies, please refer to the main paper.*

| Dataset | Metric | Baseline | Ours | Improvement |
|:---:|:---:|:---:|:---:|:---:|
| SceneFlow | EPE | x.xx | **x.xx** | - |
| KITTI 2015 | D1-all (%) | x.xx | **x.xx** | - |
| Middlebury | 2.0px error | x.xx | **x.xx** | - |

## 📁 Repository Structure

```text
.
├── core/
│   ├── edge_datasets.py      # DataLoader for geometric edge training
│   ├── edge_metrics.py       # Evaluation metrics for edges (ODS / OIS)
│   ├── edge_models.py        # Edge branch architecture (EdgeHead, Refinement, Spatial Attention)
│   ├── extractor.py          # Shared feature backbone for context and edge extraction
│   ├── geometry.py           # Geometry Encoding Volume computation
│   ├── igev_stereo.py        # Main network architecture (Stereo matching + Edge fusion modules)
│   ├── rcf_models.py         # RCF (Rich Convolutional Features) implementation for edge maps
│   ├── stereo_datasets.py    # Standard Stereo Matching datasets loading
│   └── update.py             # ConvGRU update blocks & edge-guided motion encoders
├── demo_imgs.py              # Script to predict disparity for a list of images
├── demo_video.py             # Script to run inference on a video sequence
├── eval_edge.py              # Script to evaluate edge branch
├── evaluate_stereo.py        # Script to evaluate the full stereo model (EPE, D1, etc.)
├── gtedge.py                 # Utility to generate GT edges from disparity gradients
├── train_edge.py             # Dedicated script to train the edge generation branch
├── train_stereo.py           # Main training script for the full stereo network
└── env.sh                    # Environment setup script
