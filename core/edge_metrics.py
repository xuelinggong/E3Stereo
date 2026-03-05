"""
Edge detection evaluation metrics: ODS / OIS.

ODS (Optimal Dataset Scale): F1-score when using a single optimal threshold for the entire dataset.
OIS (Optimal Image Scale): Mean F1-score when using the optimal threshold for each individual image.

Distance-threshold based matching: A predicted edge pixel is considered a correct match if it lies 
within 'dist_thresh' pixels of a GT edge.
Default dist_thresh = max(2, 0.0075 * image_diagonal), consistent with BSDS conventions.
"""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def _precision_recall_f1(pred_bin, gt_bin, dist_thresh_px):
    """
    Precision / Recall / F1 based on distance threshold.
    pred_bin, gt_bin: [H, W] binary 0/1 arrays.
    dist_thresh_px: Distance threshold for matching (pixels).
    """
    pred_bin = pred_bin.astype(np.uint8)
    gt_bin = gt_bin.astype(np.uint8)
    n_pred = pred_bin.sum()
    n_gt = gt_bin.sum()

    if n_pred == 0:
        prec = 1.0 if n_gt == 0 else 0.0
        rec = 1.0 if n_gt == 0 else 0.0
    elif n_gt == 0:
        prec = 0.0
        rec = 1.0
    else:
        # Distance to the nearest GT edge (compute EDT on background=1 to get distance to foreground)
        dist_to_gt = distance_transform_edt(1 - gt_bin)
        dist_to_pred = distance_transform_edt(1 - pred_bin)
        
        # Precision: Percentage of predicted edges that fall within dist_thresh of a GT edge
        tp_prec = (pred_bin & (dist_to_gt <= dist_thresh_px)).sum()
        prec = float(tp_prec) / float(n_pred)
        
        # Recall: Percentage of GT edges covered within dist_thresh of predicted edges
        tp_rec = (gt_bin & (dist_to_pred <= dist_thresh_px)).sum()
        rec = float(tp_rec) / float(n_gt)

    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


def ods_ois_single_image(pred_prob, gt_binary, dist_thresh_px, thresh_list=None):
    """
    ODS/OIS statistics for a single image.
    pred_prob: [H, W] float [0, 1]
    gt_binary: [H, W] 0/1 or [0, 1] continuous values (values > 0.5 are treated as edges)
    dist_thresh_px: Distance threshold (pixels)
    thresh_list: List of thresholds to sweep; defaults to 0.01~0.99 with step 0.01

    Returns:
        best_f1_ois: Optimal F1 for this image (used for OIS)
        best_thresh_ois: Optimal threshold for this image
        curve: list of (thresh, prec, rec, f1) used for ODS aggregation
    """
    gt_bin = (np.asarray(gt_binary, dtype=np.float32) > 0.5).astype(np.uint8)
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    if pred_prob.shape != gt_bin.shape:
        pred_prob = cv2.resize(
            pred_prob, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

    if thresh_list is None:
        thresh_list = np.linspace(0.01, 0.99, 99)

    curve = []
    best_f1 = 0.0
    best_thresh = 0.5
    for t in thresh_list:
        pred_bin = (pred_prob > t).astype(np.uint8)
        prec, rec, f1 = _precision_recall_f1(pred_bin, gt_bin, dist_thresh_px)
        curve.append((float(t), prec, rec, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_f1, best_thresh, curve


def compute_ods_ois(pred_list, gt_list, dist_thresh_frac=0.0075, thresh_list=None):
    """
    Aggregate ODS / OIS calculation over multiple images.
    pred_list: list of [H, W] float [0, 1]
    gt_list: list of [H, W] 0/1 or [0, 1]
    dist_thresh_frac: distance threshold = max(2, dist_thresh_frac * image_diagonal)

    Returns:
        ods: float
        ois: float
        ods_thresh: Optimal threshold corresponding to ODS
    """
    if thresh_list is None:
        thresh_list = np.linspace(0.01, 0.99, 99)

    n = len(pred_list)
    assert n == len(gt_list), "pred_list and gt_list length mismatch"

    # Aggregation by threshold: (prec, rec, f1) across all images for each threshold
    thresh_to_f1 = {t: [] for t in thresh_list}
    ois_scores = []

    for pred, gt in zip(pred_list, gt_list):
        h, w = gt.shape[:2]
        diag = np.sqrt(h * h + w * w)
        dist_px = max(2, int(round(dist_thresh_frac * diag)))

        best_f1, _, curve = ods_ois_single_image(pred, gt, dist_px, thresh_list)
        ois_scores.append(best_f1)

        for (t, prec, rec, f1) in curve:
            thresh_to_f1[t].append(f1)

    # OIS: Mean of the optimal F1 scores for each image
    ois = float(np.mean(ois_scores))

    # ODS: Optimal single threshold for the entire dataset
    best_ods = 0.0
    best_ods_thresh = 0.5
    for t in thresh_list:
        avg_f1 = np.mean(thresh_to_f1[t])
        if avg_f1 > best_ods:
            best_ods = avg_f1
            best_ods_thresh = t

    return float(best_ods), ois, best_ods_thresh
