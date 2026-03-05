#!/usr/bin/env python
"""
Independent training script for the geometric edge branch.

Reuses the Feature backbone + EdgeHead of IGEV to learn to predict geometric edges from RGB on SceneFlow
(Labels are GT edges generated from disparity gradients, see gtedge.py).

Validation goal: Can the model learn depth-discontinuity geometric edge features from a single RGB image on synthetic data.
"""
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.edge_models import GeoEdgeNet
from core.edge_datasets import fetch_edge_dataloader, fetch_edge_eval_dataloader
from core.edge_metrics import compute_ods_ois

from torch.cuda.amp import GradScaler, autocast


def run_ods_ois_eval(model, eval_loader, step, writer, dist_thresh_frac=0.0075):
    """Calculate ODS/OIS on eval_loader, write to TensorBoard and print."""
    model.eval()
    pred_list = []
    gt_list = []
    with torch.no_grad():
        for img, edge_gt in eval_loader:
            img = img.cuda()
            edge_gt = edge_gt.cuda()
            img_norm = (2 * (img / 255.0) - 1.0).contiguous()
            with autocast(enabled=True):
                edge_logits = model(img_norm)
            pred_prob = torch.sigmoid(edge_logits).squeeze(1).cpu().numpy()
            gt_np = edge_gt.squeeze(1).cpu().numpy()
            for i in range(pred_prob.shape[0]):
                pred_list.append(pred_prob[i])
                gt_list.append(gt_np[i])
    model.train()

    ods, ois, ods_thresh = compute_ods_ois(
        pred_list, gt_list, dist_thresh_frac=dist_thresh_frac
    )
    if writer is not None:
        writer.add_scalar("eval/ODS", ods, step)
        writer.add_scalar("eval/OIS", ois, step)
        writer.add_scalar("eval/ODS_thresh", ods_thresh, step)
    logging.info(
        "Eval ODS/OIS (step %d): ODS=%.4f (thresh=%.3f), OIS=%.4f",
        step, ods, ods_thresh, ois
    )
    return ods, ois


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def edge_loss(pred_logits, target, pos_weight=2.0):
    """
    BCE with pos_weight to handle class imbalance due to sparse edge pixels (few positive samples).
    target: [B, 1, H, W], range [0, 1]
    """
    target = target.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(
        pred_logits, target, pos_weight=torch.tensor(pos_weight, device=pred_logits.device)
    )


def edge_metrics(pred_logits, target, thresh=0.5):
    """Calculate precision, recall, F1 (binarization threshold 0.5)"""
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    target_bin = (target > thresh).float()
    tp = (pred * target_bin).sum()
    fp = (pred * (1 - target_bin)).sum()
    fn = ((1 - pred) * target_bin).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": prec.item(), "recall": rec.item(), "f1": f1.item()}


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, logdir, exp_name):
        self.model = model
        self.scheduler = scheduler
        self.logdir = logdir
        self.exp_name = exp_name
        self.total_steps = 0
        self.running_loss = {}
        self.best_f1 = -1.0
        self.writer = SummaryWriter(log_dir=logdir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        lr = self.scheduler.get_last_lr()[0]
        training_str = f"[{self.total_steps + 1:6d}, {lr:10.7f}] "
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        logging.info(f"Edge Training ({self.total_steps}): {training_str + metrics_str}")

        # Write to TensorBoard and maintain the best model based on F1 score
        for k in self.running_loss:
            avg_val = self.running_loss[k] / Logger.SUM_FREQ
            self.writer.add_scalar(k, avg_val, self.total_steps)
            if k == "f1":
                if avg_val > self.best_f1:
                    self.best_f1 = avg_val
                    save_path = Path(self.logdir) / f"{self.exp_name}_best.pth"
                    torch.save(self.model.state_dict(), save_path)
                    logging.info("New best F1: %.4f at step %d, saved checkpoint: %s",
                                 self.best_f1, self.total_steps, save_path)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(GeoEdgeNet(
        use_refinement=not getattr(args, "no_refinement", False),
        refine_iters=getattr(args, "refine_iters", 1),
        use_spatial_attn=not getattr(args, "no_spatial_attn", False),
    ))
    logging.info("Parameter Count: %d" % count_parameters(model))

    train_loader = fetch_edge_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy="linear"
    )
    # logger will automatically save the best model based on f1 to {logdir}/{name}_best.pth
    logger = Logger(model, scheduler, args.logdir, args.name)

    save_freq = getattr(args, "save_freq", 0)  # 0=no regular saving, only save best and final

    total_steps = 0
    if args.restore_ckpt:
        logging.info("Loading checkpoint: %s", args.restore_ckpt)
        ckpt = torch.load(args.restore_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=True)
            if "optimizer" in ckpt and ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            total_steps = ckpt.get("total_steps", 0)
            logging.info("Checkpoint loaded (resumed from step %d).", total_steps)
        else:
            model.load_state_dict(ckpt, strict=True)
            logging.info("Checkpoint loaded (state_dict only).")

    model.cuda()
    model.train()
    scaler = GradScaler(enabled=args.mixed_precision)
    pos_weight = getattr(args, "pos_weight", 2.0)

    eval_loader = None
    if getattr(args, "eval_freq", 0) > 0:
        eval_loader = fetch_edge_eval_dataloader(args, max_samples=getattr(args, "eval_samples", 500))

    for epoch in range(999):
        for img, edge_gt in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img = img.cuda()
            edge_gt = edge_gt.cuda()

            # Normalize input to [-1, 1], consistent with IGEV stereo
            img_norm = (2 * (img / 255.0) - 1.0).contiguous()

            optimizer.zero_grad()
            with autocast(enabled=args.mixed_precision):
                edge_logits = model(img_norm)
                loss = edge_loss(edge_logits, edge_gt, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                metrics = edge_metrics(edge_logits, edge_gt)
                metrics["loss"] = loss.item()
            logger.push(metrics)
            total_steps += 1

            # Periodic ODS/OIS evaluation
            if eval_loader is not None and total_steps % args.eval_freq == 0 and total_steps > 0:
                run_ods_ois_eval(
                    model, eval_loader, total_steps, logger.writer,
                    dist_thresh_frac=getattr(args, "ods_dist_frac", 0.0075)
                )

            # Periodically save checkpoint (including optimizer/scheduler for easy resume)
            if save_freq > 0 and total_steps % save_freq == 0 and total_steps > 0:
                ckpt_path = Path(args.logdir) / f"{args.name}_step{total_steps}.pth"
                torch.save({
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "total_steps": total_steps,
                }, ckpt_path)
                logging.info("Saved checkpoint: %s", ckpt_path)

            if total_steps >= args.num_steps:
                break

        if total_steps >= args.num_steps:
            break

    # Perform a final ODS/OIS evaluation before ending training
    if eval_loader is not None:
        run_ods_ois_eval(
            model, eval_loader, total_steps, logger.writer,
            dist_thresh_frac=getattr(args, "ods_dist_frac", 0.0075)
        )

    logging.info("FINISHED TRAINING")
    logger.close()
    save_path = Path(args.logdir) / f"{args.name}.pth"
    torch.save(model.state_dict(), save_path)
    logging.info("Saved: %s", save_path)
    return str(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train geometric edge branch on SceneFlow")
    parser.add_argument("--name", default="geo-edge", help="experiment name")
    parser.add_argument("--restore_ckpt", default=None, help="resume from checkpoint")
    parser.add_argument("--logdir", default="./checkpoints/geo-edge", help="log and checkpoint dir")
    parser.add_argument("--data_root", default="./data/sceneflow", help="SceneFlow root")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--pos_weight", type=float, default=2.0, help="BCE pos_weight for class imbalance")
    parser.add_argument("--image_size", type=int, nargs="+", default=[320, 736])
    parser.add_argument("--spatial_scale", type=float, nargs="+", default=[-0.2, 0.4])
    parser.add_argument("--saturation_range", type=float, nargs="+", default=[0.6, 1.4])
    parser.add_argument("--noyjitter", action="store_true")
    parser.add_argument("--eval_freq", type=int, default=10000,
                        help="Evaluate ODS/OIS every N steps, 0 means no evaluation")
    parser.add_argument("--save_freq", type=int, default=10000,
                        help="Save checkpoint every N steps (includes optimizer/scheduler to resume), 0 means no periodic saving")
    parser.add_argument("--eval_samples", type=int, default=500,
                        help="Maximum number of samples used for ODS/OIS evaluation")
    parser.add_argument("--ods_dist_frac", type=float, default=0.0075,
                        help="ODS/OIS distance threshold = max(2, frac*image_diagonal)")
    parser.add_argument("--no_refinement", action="store_true",
                        help="Disable EdgeRefinementModule (used when loading old ckpts)")
    parser.add_argument("--no_spatial_attn", action="store_true",
                        help="Disable SpatialAttention (used when loading old ckpts)")
    parser.add_argument("--refine_iters", type=int, default=1,
                        help="Number of refine iterations, 1=single pass, 2/3=iterative sharpening (shares the same Refine module)")
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        filename=f"{args.logdir}/train{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    train(args)
