"""
Generate ground-truth geometric edge maps from SceneFlow disparity.

Geometric edges are defined as pixels where the disparity gradient exceeds a threshold,
corresponding to depth discontinuities. These GT edge maps serve as supervision labels
for the GeoEdgeNet branch and as auxiliary inputs to the edge-guided stereo matching pipeline.

Supported gradient modes:
  - sobel:          Sobel gradient magnitude (default)
  - blur_sobel:     Gaussian blur first, then Sobel (retains large-scale geometric edges only)
  - laplacian:      Laplacian edge detection
  - laplacian_close: Laplacian + morphological closing to fill gaps
  - canny:          Canny edge detector (requires canny_low/canny_high thresholds)
  - sobel_rel:      Relative Sobel gradient: |grad| / (|disp| + eps)
"""

import os
import os.path as osp
from glob import glob
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def read_pfm(file):
    """Minimal PFM reader compatible with SceneFlow disparity (.pfm)."""
    import re

    with open(file, "rb") as f:
        header = f.readline().rstrip().decode("ascii")
        if header not in ("PF", "Pf"):
            raise ValueError("Not a PFM file: %s" % file)

        dim_line = f.readline().decode("ascii")
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("ascii")
        width, height = map(int, re.findall(r"\d+", dim_line))

        scale = float(f.readline().rstrip().decode("ascii"))
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        data = np.reshape(data, (height, width, 3) if header == "PF" else (height, width))
        data = np.flipud(data)
    return data


def _grad_mag(disp, mode="sobel", blur_ksize=5, blur_sigma=1.0):
    """Compute gradient magnitude of a disparity map."""
    disp = np.asarray(disp, np.float32)

    # Set invalid values to zero to avoid generating false edges.
    invalid = ~np.isfinite(disp)
    disp[invalid] = 0.0

    # Optional: Apply Gaussian blur first to remove high-frequency noise or fine textures.
    if mode in ("blur_sobel", "blur_laplacian"):
        # Gaussian blur, retaining only large-scale structures.
        k = int(blur_ksize)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1  # kernel size must be odd
        disp = cv2.GaussianBlur(disp, (k, k), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

    if mode in ("sobel", "blur_sobel"):
        # First-order gradient (Slope): Planar slopes will be non-zero constants -> prone to whitening out entire regions
        gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
    elif mode in ("laplacian", "blur_laplacian", "laplacian_close"):
        # Second-order gradient (Curvature): Second derivative of planar slopes is close to 0, responding only at curvatures/jumps
        lap = cv2.Laplacian(disp, cv2.CV_32F, ksize=3)
        mag = np.abs(lap)
    else:
        raise ValueError(f"Unknown mode for _grad_mag: {mode}")

    return mag


def disp_to_edge(
    disp,
    grad_thresh=0.5,
    mode="sobel",
    blur_ksize=5,
    blur_sigma=1.0,
    canny_low=None,
    canny_high=None,
):
    """
    Generate geometric edges based on disparity gradients:
    - mode="sobel"              : Direct Sobel, absolute first-order gradient
    - mode="sobel_rel"          : Sobel + relative gradient threshold: |∇d| / (|d|+eps), friendlier to large planes
    - mode="blur_sobel"         : Blur then Sobel, retaining only large-scale geometric edges
    - mode="laplacian"          : Direct second derivative (Laplacian), less sensitive to planar slopes
    - mode="blur_laplacian"     : Blur then Laplacian, focusing more on large-scale curvatures/jumps
    - mode="laplacian_close"    : Direct second derivative + morphological closing (dilate then erode) to thicken and fill holes
    - mode="canny"              : Normalized disparity + Canny edge detection
    - Generate binary edges based on the gradient magnitude threshold grad_thresh
    """
    disp_arr = np.asarray(disp, np.float32)

    # Specific Canny branch (bypasses _grad_mag)
    if mode == "canny":
        # Handle invalid values
        d = np.copy(disp_arr)
        invalid = ~np.isfinite(d)
        d[invalid] = 0.0

        # Robust normalization to [0, 255] based on percentiles
        vmin, vmax = np.percentile(d, 1), np.percentile(d, 99)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        d = np.clip(d, vmin, vmax)
        d = (d - vmin) / (vmax - vmin) * 255.0
        d8 = d.astype(np.uint8)

        # Optional blur to reduce noise
        if blur_ksize and blur_ksize > 1:
            k = int(blur_ksize)
            if k % 2 == 0:
                k += 1
            d8 = cv2.GaussianBlur(d8, (k, k), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

        # Canny thresholds: use relatively conservative defaults if not explicitly provided
        if canny_low is None and canny_high is None:
            low = 50
            high = 150
        else:
            low = canny_low if canny_low is not None else 50
            high = canny_high if canny_high is not None else low * 3.0

        edge = cv2.Canny(d8, threshold1=float(low), threshold2=float(high))
        return edge

    # Map to base gradient mode
    base_mode = mode
    if mode == "laplacian_close":
        base_mode = "laplacian"
    if mode == "sobel_rel":
        base_mode = "sobel"

    mag = _grad_mag(disp_arr, mode=base_mode, blur_ksize=blur_ksize, blur_sigma=blur_sigma)

    # Relative gradient threshold: |∇d| / (|d| + eps)
    if mode == "sobel_rel":
        denom = np.abs(disp_arr)
        eps = 1e-3
        denom[denom < eps] = eps
        mag = mag / denom
    edge = (mag >= grad_thresh).astype(np.uint8)

    if mode == "laplacian_close":
        # Morphological closing: dilate then erode to fill gaps and thicken thin edges
        k = int(blur_ksize)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.dilate(edge, kernel, iterations=1)
        edge = cv2.erode(edge, kernel, iterations=1)

    edge = (edge > 0).astype(np.uint8) * 255
    return edge


def process_sceneflow_disp(
    root_disp="./data/sceneflow/disparity",
    root_edge="./data/sceneflow/ggtedge",
    split_list=("TRAIN", "TEST"),
    grad_thresh=2.5,
    mode="sobel",
    blur_ksize=5,
    blur_sigma=1.0,
    canny_low=None,
    canny_high=None,
    one_per_folder=False,
    max_files_per_split=None,
):
    """
    Iterate over SceneFlow disparity directories to generate GT Edge Maps.

    Directory structure:
        /data/sceneflow/
            disparity/      # Disparity .pfm
                TRAIN/...
                TEST/...
            frames_finalpass/  # RGB images
                TRAIN/...
                TEST/...
            gtedge/         # The GT edges to be generated (output of this script)
                TRAIN/...
                TEST/...
    root_disp corresponds to disparity above, root_edge corresponds to gtedge.
    """

    for split in split_list:
        # example：.../disparity/TRAIN/**.pfm
        pattern = osp.join(root_disp, split, "**", "*.pfm")
        all_files = sorted(glob(pattern, recursive=True))

        # Select the list of files to process
        if one_per_folder:
            # Take only one image per directory for quick visualization/validation
            disp_files = []
            seen_dirs = set()
            for p in all_files:
                ddir = osp.dirname(p)
                if ddir in seen_dirs:
                    continue
                seen_dirs.add(ddir)
                disp_files.append(p)
        else:
            disp_files = list(all_files)

        # Further limit the maximum number of processed files per split
        if max_files_per_split is not None and max_files_per_split > 0:
            disp_files = disp_files[: max_files_per_split]

        print(
            f"[gtedge] Split={split}, found {len(all_files)} disparity files, "
            f"selected {len(disp_files)} for processing "
            f"(one_per_folder={one_per_folder}, max_files_per_split={max_files_per_split})"
        )

        for disp_path in tqdm(disp_files, desc=f"GT Edge {split}", ncols=80):
            # Read disparity
            disp = read_pfm(disp_path)
            if disp.ndim == 3:
                # Some PFMs might be 3-channel; take the first channel here
                disp = disp[..., 0]

            edge = disp_to_edge(
                disp,
                grad_thresh=grad_thresh,
                mode=mode,
                blur_ksize=blur_ksize,
                blur_sigma=blur_sigma,
                canny_low=canny_low,
                canny_high=canny_high,
            )

            # Generate corresponding output paths: replace root_disp with root_edge and change the extension to .png
            # Example:
            #   disp_path = .../disparity/TRAIN/A/xxx/0000.pfm
            #   rel_path = TRAIN/A/xxx/0000.pfm
            #   edge_path = .../gtedge/TRAIN/A/xxx/0000.png
            rel_path = osp.relpath(disp_path, root_disp)
            edge_rel = rel_path.replace(".pfm", ".png")
            edge_path = osp.join(root_edge, edge_rel)

            os.makedirs(osp.dirname(edge_path), exist_ok=True)
            cv2.imwrite(edge_path, edge)


def process_single_dir(
    disp_dir,
    root_disp="./data/sceneflow/disparity",
    root_edge="./data/sceneflow/ggtedge",
    grad_thresh=2.5,
    mode="sobel",
    blur_ksize=5,
    blur_sigma=1.0,
    canny_low=None,
    canny_high=None,
):
    """
    Process only a specific disparity subdirectory (e.g.,
        /path/to/data/sceneflow/disparity/TRAIN/15mm_focallength/scene_backwards/fast/left
    ) for quick debugging and visualization.
    """
    disp_dir = osp.abspath(disp_dir)
    pattern = osp.join(disp_dir, "*.pfm")
    disp_files = sorted(glob(pattern))

    print(f"[gtedge] Single dir mode: {disp_dir}, found {len(disp_files)} disparity files")

    for disp_path in tqdm(disp_files, desc="GT Edge single dir", ncols=80):
        disp = read_pfm(disp_path)
        if disp.ndim == 3:
            disp = disp[..., 0]

        edge = disp_to_edge(
            disp,
            grad_thresh=grad_thresh,
            mode=mode,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
            canny_low=canny_low,
            canny_high=canny_high,
        )

        rel_path = osp.relpath(disp_path, root_disp)
        edge_rel = rel_path.replace(".pfm", ".png")
        edge_path = osp.join(root_edge, edge_rel)

        os.makedirs(osp.dirname(edge_path), exist_ok=True)
        cv2.imwrite(edge_path, edge)


if __name__ == "__main__":
    """
    Usage examples:
        # 1) Direct Sobel + threshold 2.5 (equivalent to previous versions)
        python gtedge.py --grad_thresh 2.5 --mode sobel

        # 2) Gaussian blur then gradient, retaining only large-scale geometric edges
        python gtedge.py --grad_thresh 2.5 --mode blur_sobel --blur_ksize 5 --blur_sigma 1.0
    """

    parser = argparse.ArgumentParser(description="Generate GT geometric edge maps from SceneFlow disparity.")
    parser.add_argument("--grad_thresh", type=float, default=3.0, help="梯度幅值阈值，越大边越稀疏（单位：像素差）。")
    parser.add_argument(
        "--mode",
        type=str,
        default="sobel",
        choices=["sobel", "sobel_rel", "blur_sobel", "laplacian", "blur_laplacian", "laplacian_close", "canny"],
        help=(
            "edge 计算模式："
            "sobel（原始一阶绝对梯度）, "
            "sobel_rel（一阶相对梯度 |∇d|/(|d|+eps)，对大平面更友好）, "
            "blur_sobel（先模糊再一阶）, "
            "laplacian（原始二阶）, "
            "blur_laplacian（先模糊再二阶，对平面斜坡最不敏感）, "
            "laplacian_close（二阶 + 闭运算：先膨胀再腐蚀，加粗并连通细边）, "
            "canny（在 disparity 上做归一化后的 Canny）"
        ),
    )
    parser.add_argument(
        "--blur_ksize",
        type=int,
        default=5,
        help="高斯模糊核大小（用于 blur_* 模式和 canny 的预平滑，需为奇数）。",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=1.0,
        help="高斯模糊 sigma（用于 blur_* 模式和 canny 的预平滑）。",
    )
    parser.add_argument(
        "--canny_low",
        type=float,
        default=None,
        help="Canny 低阈值（0-255），留空则使用默认50。",
    )
    parser.add_argument(
        "--canny_high",
        type=float,
        default=None,
        help="Canny 高阈值（0-255），留空则使用 max(canny_low*3,150)。",
    )
    parser.add_argument(
        "--one_per_folder",
        action="store_true",
        help="每个 disparity 子目录只处理一张，用于快速可视化/验证。",
    )
    parser.add_argument(
        "--max_files_per_split",
        type=int,
        default=0,
        help="每个 split 最多处理多少张（按路径排序截断），默认50；设为<=0表示不截断。",
    )
    parser.add_argument(
        "--disp_dir",
        type=str,
        default=None,
        help="仅处理指定 disparity 子目录（例如 TRAIN/15mm_focallength/.../left），用于快速调试。",
    )

    args = parser.parse_args()

    if args.disp_dir is not None:
        process_single_dir(
            disp_dir=args.disp_dir,
            grad_thresh=args.grad_thresh,
            mode=args.mode,
            blur_ksize=args.blur_ksize,
            blur_sigma=args.blur_sigma,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )
    else:
        process_sceneflow_disp(
            grad_thresh=args.grad_thresh,
            mode=args.mode,
            blur_ksize=args.blur_ksize,
            blur_sigma=args.blur_sigma,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            one_per_folder=args.one_per_folder,
            max_files_per_split=args.max_files_per_split,
        )

