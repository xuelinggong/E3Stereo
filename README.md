# Edge-Guided Iterative Geometry Encoding Volume for Stereo Matching

The paper proposes a comprehensive edge-guided stereo matching framework built on top of IGEV-Stereo. We introduce a lightweight geometric edge branch (**GeoEdgeNet**) that predicts depth-discontinuity edge maps from RGB images, and systematically study how these edge priors can be injected into multiple stages of the stereo pipeline — including cost volume construction, context fusion, GRU update, disparity upsampling, and post-processing refinement — using three flexible fusion mechanisms: **concat**, **FiLM**, and **gated**.

---

## Method Overview

Our framework extends [IGEV-Stereo (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf) with modular edge guidance at six injection points:

| Module | Flag | Description |
|---|---|---|
| **Edge-Guided GWC** | `--edge_guided_gwc` | Injects edge into GWC correlation feature attention for boundary-aware initial cost |
| **Edge-Guided Cost Agg** | `--edge_guided_cost_agg` | Injects edge into Hourglass FeatureAtt for better `init_disp` at boundaries |
| **Edge Context Fusion** | `--edge_context_fusion` | Fuses edge into GRU context features |
| **Edge-Guided Disp Head** | `--edge_guided_disp_head` | Guides `delta_disp` prediction in the GRU update step |
| **Edge-Guided Upsampling** | `--edge_guided_upsample` | Guides sub-pixel disparity upsampling for sharper boundaries |
| **Edge-Guided Refinement** | `--edge_guided_refinement` | Post-processing residual refinement at boundaries |

Each module supports **concat**, **FiLM**, and **gated** fusion (selectable via `--edge_*_fusion_mode`).

---

## Datasets

| Dataset | Split | Link |
|---|---|---|
| **Scene Flow** (FlyingThings3D + Monkaa + Driving) | Train / Test | [https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) |
| **KITTI 2012** | Train / Test | [http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) |
| **KITTI 2015** | Train / Test | [http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) |
| **Middlebury** | TrainingH / TrainingF / TrainingQ | [https://vision.middlebury.edu/stereo/submit3/](https://vision.middlebury.edu/stereo/submit3/) |
| **ETH3D** | two_view_training | [https://www.eth3d.net/datasets#low-res-two-view-test-data](https://www.eth3d.net/datasets#low-res-two-view-test-data) |

---

## Main Results

Results on SceneFlow (Things, TEST split) after Scene Flow pre-training:

| Method | Edge Module | SceneFlow EPE | SceneFlow D1 |
|---|---|---|---|
| IGEV-Stereo (baseline) | — | 0.47 | 0.72% |
| + Edge-Guided GWC (film) | GWC | — | — |
| + Edge-Guided Upsampling (film) | Upsample | — | — |
| + Edge Context Fusion (film) | Context | — | — |
| + Combined (all modules) | Full | — | — |



## Repository Structure

```
.
├── core/
│   ├── igev_stereo.py         # Main IGEVStereo model with all edge-guidance modules
│   ├── update.py              # GRU update block (BasicMultiUpdateBlock, BasicMotionEncoder)
│   ├── extractor.py           # Feature backbone (Feature, MultiBasicEncoder)
│   ├── geometry.py            # Combined Geometry Encoding Volume
│   ├── submodule.py           # Shared convolution utilities, FeatureAtt, etc.
│   ├── edge_models.py         # GeoEdgeNet: Feature backbone + EdgeHead + refinement
│   ├── edge_datasets.py       # SceneFlow edge dataset loader (RGB + GT edge pairs)
│   ├── edge_metrics.py        # ODS/OIS evaluation metrics for edge prediction
│   ├── stereo_datasets.py     # All stereo dataset loaders (SceneFlow, KITTI, ETH3D, etc.)
│   └── utils/
│       ├── frame_utils.py     # PFM / KITTI / Middlebury / ETH3D disparity readers
│       ├── utils.py           # InputPadder and misc utilities
│       └── augmentor.py       # Data augmentation (FlowAugmentor, SparseFlowAugmentor)
├── train_stereo.py            # Main training script for edge-guided stereo matching
├── evaluate_stereo.py         # Evaluation script (SceneFlow, KITTI, ETH3D, Middlebury)
├── train_edge.py              # Standalone training script for GeoEdgeNet
├── eval_edge.py               # Evaluation and visualization script for GeoEdgeNet
├── gtedge.py                  # Generate GT geometric edge maps from SceneFlow disparity
├── save_disp.py               # Export disparity maps for KITTI benchmark submission
├── demo_imgs.py               # Demo: run stereo on image pairs and save colorized disparity
├── demo_video.py              # Demo: run stereo on a left/right image sequence as video
├── train.sh                   # Example training shell script
├── train_edge.sh              # Shell script for standalone GeoEdgeNet training
├── eval_edge.sh               # Shell script for GeoEdgeNet evaluation & visualization
├── env.sh                     # Environment setup (standard)
└── env_bfloat16.sh            # Environment setup (bfloat16-compatible PyTorch)
```

### Key File Descriptions

**`core/igev_stereo.py`** — The central model. Contains all six edge-guidance injection points as optional modules controlled by `args` flags. When no edge flags are set, the model is identical to original IGEV-Stereo.

**`core/edge_models.py`** — `GeoEdgeNet`: a lightweight edge predictor that reuses the IGEV `Feature` backbone, adds an `EdgeHead` with multi-scale fusion, spatial attention, and an iterative `EdgeRefinementModule` for sharp thin-edge prediction.

**`core/update.py`** — GRU update block. `BasicMotionEncoder` supports optional edge injection (concat/FiLM/gated). `BasicMultiUpdateBlock` supports edge-guided `delta_disp` prediction.

**`gtedge.py`** — Preprocessing utility: traverses the SceneFlow disparity directory and generates GT geometric edge PNG maps using configurable gradient modes (Sobel, blur-Sobel, Laplacian, Canny).

**`train_edge.py`** — Standalone training for `GeoEdgeNet` on SceneFlow, using disparity-derived GT edges. Includes periodic ODS/OIS evaluation.

**`evaluate_stereo.py`** — Full evaluation on all benchmarks; computes EPE, D1, edge-EPE, flat-EPE, and writes per-experiment result files.

---

## Environment Setup

```bash
conda create -n igev-edge python=3.8
conda activate igev-edge
bash env.sh
```

For bfloat16 training support:
```bash
bash env_bfloat16.sh
```

---

## Data Preparation

### Step 1 — Download datasets

Place datasets under `/data/` following this layout:

```
/data/
├── sceneflow/
│   ├── frames_finalpass/
│   │   ├── TRAIN/  (A/, B/, C/, 15mm_focallength/, ...)
│   │   └── TEST/
│   └── disparity/
│       ├── TRAIN/
│       └── TEST/
├── KITTI/
│   ├── KITTI_2012/  (training/, testing/)
│   └── KITTI_2015/  (training/, testing/)
├── Middlebury/
│   ├── trainingH/
│   ├── trainingF/
│   └── trainingQ/
└── ETH3D/
    ├── two_view_training/
    └── two_view_training_gt/
```

### Step 2 — Generate GT edge maps (required for `--edge_source gt`)

```bash
python gtedge.py \
    --root_disp /data/sceneflow/disparity \
    --root_edge /data/sceneflow/gtedge \
    --splits TRAIN TEST \
    --grad_thresh 2.5 \
    --mode sobel
```

This will generate `gtedge/TRAIN/...` and `gtedge/TEST/...` PNG edge maps under the SceneFlow root.

---

## Training

### Train stereo model on SceneFlow (with edge-guided GWC, GT edge source)

```bash
python train_stereo.py \
    --name igev-edge-gwc \
    --logdir ./checkpoints/igev-edge-gwc \
    --train_datasets sceneflow \
    --lr 0.0002 \
    --num_steps 200000 \
    --batch_size 8 \
    --edge_source gt \
    --edge_model None \
    --edge_guided_gwc \
    --edge_gwc_fusion_mode film
```

### Fine-tune on KITTI

```bash
python train_stereo.py \
    --name igev-edge-kitti \
    --logdir ./checkpoints/igev-edge-kitti \
    --restore_ckpt ./checkpoints/igev-edge-gwc/200000_igev-edge-gwc.pth \
    --train_datasets kitti \
    --lr 0.0001 \
    --num_steps 50000 \
    --batch_size 6 \
    --edge_source gt \
    --edge_model None \
    --edge_guided_gwc \
    --edge_gwc_fusion_mode film
```

### Train GeoEdgeNet (geometric edge branch)

```bash
# Step 1: generate GT edges (see above)
# Step 2: train
bash train_edge.sh
# or equivalently:
python train_edge.py \
    --name geo-edge \
    --logdir ./checkpoints/geo-edge \
    --data_root /data/sceneflow \
    --batch_size 16 \
    --lr 0.0001 \
    --num_steps 50000 \
    --mixed_precision \
    --refine_iters 3
```

---

## Evaluation

### Evaluate on SceneFlow

```bash
python evaluate_stereo.py \
    --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth \
    --dataset sceneflow \
    --edge_source gt
```

### Evaluate on KITTI

```bash
python evaluate_stereo.py \
    --restore_ckpt ./pretrained_models/kitti/kitti15.pth \
    --dataset kitti
```

### Evaluate on ETH3D / Middlebury

```bash
python evaluate_stereo.py \
    --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth \
    --dataset eth3d

python evaluate_stereo.py \
    --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth \
    --dataset middlebury_H
```

### KITTI benchmark submission

```bash
python save_disp.py \
    --restore_ckpt ./pretrained_models/kitti/kitti15.pth \
    --output_directory output/kitti2015/disp_0
```

---

## Demo

```bash
# Run on image pairs and save colorized disparity
python demo_imgs.py \
    --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth \
    -l /path/to/left_imgs/*.png \
    -r /path/to/right_imgs/*.png \
    --output_directory ./demo-output/

# Run on a synchronized left/right image sequence (video)
python demo_video.py \
    --restore_ckpt ./pretrained_models/kitti/kitti15.pth \
    -l /path/to/left_imgs/*.png \
    -r /path/to/right_imgs/*.png
```

---

## Bfloat16 Training

If you encounter NaN during float16 training (overflow), switch to bfloat16:

```bash
python train_stereo.py --mixed_precision --precision_dtype bfloat16 [other args]
```

---

## Acknowledgements

This project builds upon [IGEV-Stereo (CVPR 2023)](https://github.com/gangweiX/IGEV) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo). We thank the authors for their excellent open-source contributions.
