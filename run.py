import os
import glob
import math
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from net.model import WDNet
from utils.val_utils import compute_psnr_ssim  # PSNR/SSIM utility
from torchvision.transforms import functional as TF
from utils.swt_loss_simple import SWTLoss

import lpips


# ---------------------------
# Helper Functions
# ---------------------------
def list_images(folder: str):
    """List image files in a folder."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files


def pair_by_name(src_paths: List[str], gt_root: str) -> List[Tuple[str, str]]:
    """Match source and ground truth images by name."""
    pairs = []
    gt_map = {os.path.basename(p): p for p in list_images(gt_root)}

    for sp in src_paths:
        name = os.path.basename(sp)

        cand = name.replace("rain-", "norain-")
        if cand in gt_map:
            pairs.append((sp, gt_map[cand])); continue

        if name.startswith("rain"):
            cand2 = name.replace("rain", "norain", 1)
            if cand2 in gt_map:
                pairs.append((sp, gt_map[cand2])); continue

        if name in gt_map:
            pairs.append((sp, gt_map[name])); continue

    return pairs


def to_tensor_uint8(img: Image.Image) -> torch.Tensor:
    """Convert image to tensor."""
    return TF.to_tensor(img)


def crop_to_multiple_of(t: torch.Tensor, base: int = 16) -> torch.Tensor:
    """Crop tensor to be a multiple of a given base."""
    c, h, w = t.shape
    h2 = (h // base) * base
    w2 = (w // base) * base
    if h2 == 0 or w2 == 0:
        pad_h = (base - h % base) % base
        pad_w = (base - w % base) % base
        t = torch.nn.functional.pad(t.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        c, h, w = t.shape
        h2 = (h // base) * base
        w2 = (w // base) * base
    return t[:, :h2, :w2]


def rand_crop_pair(t1: torch.Tensor, t2: torch.Tensor, patch: int):
    """Randomly crop a patch from two tensors."""
    _, H, W = t1.shape
    if H < patch or W < patch:
        pad_h = max(0, patch - H)
        pad_w = max(0, patch - W)
        if pad_h > 0 or pad_w > 0:
            t1 = torch.nn.functional.pad(t1.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            t2 = torch.nn.functional.pad(t2.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        _, H, W = t1.shape
    top = np.random.randint(0, H - patch + 1)
    left = np.random.randint(0, W - patch + 1)
    t1c = t1[:, top:top+patch, left:left+patch]
    t2c = t2[:, top:top+patch, left:left+patch]
    return t1c, t2c


def ensure_dir_exists(path: str, must_contain: Optional[List[str]] = None) -> None:
    """Ensure directory exists and contains required subdirectories."""
    if not os.path.isdir(path):
        raise RuntimeError(f"Directory does not exist: {path}")
    if must_contain:
        missing = [d for d in must_contain if not os.path.isdir(os.path.join(path, d))]
        if missing:
            raise RuntimeError(f"Missing subdirectories: {missing}")


# ---------------------------
# Datasets
# ---------------------------
class LowLightTrainDataset(Dataset):
    def __init__(self, train_root: str, patch_size: int = 128, repeat_factor: int = 120):
        """Train dataset for low-light image enhancement."""
        super().__init__()
        ensure_dir_exists(train_root, ["rainy", "gt"])
        self.patch_size = patch_size

        input_dir = os.path.join(train_root, "rainy")
        gt_dir    = os.path.join(train_root, "gt")

        srcs = list_images(input_dir)
        self.base_pairs = pair_by_name(srcs, gt_dir)
        if len(self.base_pairs) == 0:
            raise RuntimeError(f"No matching training samples found in {input_dir} and {gt_dir}.")

        self.repeat_factor = max(1, int(repeat_factor))
        self.pairs = self.base_pairs * self.repeat_factor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, gt_path = self.pairs[idx]
        inp = Image.open(src_path).convert('RGB')
        gt  = Image.open(gt_path).convert('RGB')

        t_inp, t_gt = to_tensor_uint8(inp), to_tensor_uint8(gt)

        t_inp = crop_to_multiple_of(t_inp, base=16)
        t_gt  = crop_to_multiple_of(t_gt,  base=16)

        if np.random.rand() < 0.5:
            t_inp = TF.hflip(t_inp); t_gt = TF.hflip(t_gt)
        if np.random.rand() < 0.5:
            t_inp = TF.vflip(t_inp); t_gt = TF.vflip(t_gt)
        k = np.random.randint(0, 4)
        if k:
            t_inp = torch.rot90(t_inp, k, [1, 2])
            t_gt  = torch.rot90(t_gt, k, [1, 2])

        t_inp, t_gt = rand_crop_pair(t_inp, t_gt, self.patch_size)

        return os.path.basename(src_path), t_inp, t_gt


class LowLightValDataset(Dataset):
    def __init__(self, test_root: str):
        """Validation dataset for low-light image enhancement."""
        super().__init__()
        ensure_dir_exists(test_root, ["testA", "testB"])

        input_dir = os.path.join(test_root, "testA")
        gt_dir    = os.path.join(test_root, "testB")

        srcs = list_images(input_dir)
        self.pairs = pair_by_name(srcs, gt_dir)
        if len(self.pairs) == 0:
            raise RuntimeError(f"No matching validation samples found in {input_dir} and {gt_dir}.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, gt_path = self.pairs[idx]
        inp = Image.open(src_path).convert('RGB')
        gt  = Image.open(gt_path).convert('RGB')
        t_inp, t_gt = to_tensor_uint8(inp), to_tensor_uint8(gt)

        t_inp = crop_to_multiple_of(t_inp, base=16)
        t_gt  = crop_to_multiple_of(t_gt,  base=16)

        name = os.path.splitext(os.path.basename(src_path))[0]
        return name, t_inp, t_gt


# ---------------------------
# Lightning Model
# ---------------------------
class LowLightModule(pl.LightningModule):
    def __init__(self, lr: float = 2e-4, use_lpips: bool = True, save_images_dir: str = None,
                 w_charb=1.0, w_ssim=0.2, w_grad=0.05, w_swt=0.05, w_lpips_max=0.2,
                 lpips_warmup_steps=5000, lpips_ramp_steps=15000):
        super().__init__()
        self.save_hyperparameters(ignore=['save_images_dir'])
        self.net = WDNet()

        self.crit = CompositeLoss(lambda_charb=w_charb, lambda_ssim=w_ssim,
                                  lambda_grad=w_grad, lambda_swt=w_swt, lambda_lpips=0.0)
        self.use_lpips = bool(use_lpips)
        if self.use_lpips:
            self.lpips_metric = lpips.LPIPS(net='alex')
            self.lpips_metric.eval()
            for p in self.lpips_metric.parameters():
                p.requires_grad = False
            self.crit.set_lpips(self.lpips_metric)

        self.save_images_dir = save_images_dir
        self._val_psnr, self._val_ssim, self._val_lpips = [], [], []
        self.best_psnr = float("-inf")
        self.best_ssim = float("-inf")
        self.best_lpips = float("inf")
        self.best_epoch = -1

        self.lpips_warmup_steps = int(lpips_warmup_steps)
        self.lpips_ramp_steps   = int(lpips_ramp_steps)
        self.w_lpips_max        = float(w_lpips_max)

    def setup(self, stage=None):
        if self.use_lpips:
            self.lpips_metric.to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.net(x)

    def _lpips_weight(self, step: int) -> float:
        """Calculate LPIPS weight based on training step."""
        if not self.use_lpips:
            return 0.0
        if step < self.lpips_warmup_steps:
            return 0.0
        t = min(1.0, (step - self.lpips_warmup_steps) / max(1, self.lpips_ramp_steps))
        return self.w_lpips_max * (0.5 * (1 - math.cos(math.pi * t)))

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        pred = self.net(x)
        cur_w_lp = self._lpips_weight(self.global_step)
        total, comp = self.crit(pred, y, lpips_weight_override=cur_w_lp)

        self.log("train/total", total, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        name, x, y = batch
        pred = self.net(x)

        psnr, ssim, _ = compute_psnr_ssim(pred, y)
        self.log("val/psnr", psnr, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("val/ssim", ssim, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self._val_psnr.append(float(psnr))
        self._val_ssim.append(float(ssim))

        if self.use_lpips:
            lp = self._lpips_eval(pred, y)
            self.log("val/lpips", lp, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
            self._val_lpips.append(float(lp))

        if self.save_images_dir and (batch_idx < 5):
            os.makedirs(self.save_images_dir, exist_ok=True)
            _name = name[0] if isinstance(name, (list, tuple)) else name
            out = (pred.clamp(0, 1) * 255.0).byte().cpu().squeeze(0).permute(1, 2, 0).numpy()
            Image.fromarray(out).save(os.path.join(self.save_images_dir, f"{_name}_pred.png"))

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        psnr_mean = float(np.mean(self._val_psnr)) if self._val_psnr else 0.0
        ssim_mean = float(np.mean(self._val_ssim)) if self._val_ssim else 0.0
        lpips_mean = float(np.mean(self._val_lpips)) if self._val_lpips else 0.0
        print(f"[VAL][epoch {self.current_epoch}] PSNR: {psnr_mean:.2f} | SSIM: {ssim_mean:.4f} | LPIPS: {lpips_mean:.4f}")

        improved = False
        if psnr_mean > self.best_psnr:
            self.best_psnr = psnr_mean; improved = True
        if ssim_mean > self.best_ssim:
            self.best_ssim = ssim_mean; improved = True
        if self.use_lpips and (lpips_mean < self.best_lpips):
            self.best_lpips = lpips_mean; improved = True
        if improved:
            self.best_epoch = int(self.current_epoch)


# ---------------------------
# Main Workflow
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, help="Path to training data")
    parser.add_argument("--test_root", type=str, help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=1266, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size for crops")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
    parser.add_argument("--repeat_factor", type=int, default=1, help="Data repeat factor")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="32", choices=["16", "32"], help="Precision")
    parser.add_argument("--ckpt_dir", type=str, default="train_ckpt", help="Checkpoint directory")
    parser.add_argument("--save_images_dir", type=str, default="val_vis", help="Directory to save validation images")

    args = parser.parse_args()

    pl.seed_everything(0, workers=True)

    train_set = LowLightTrainDataset(args.train_root, args.patch_size, repeat_factor=args.repeat_factor)
    val_set   = LowLightValDataset(args.test_root)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    model = LowLightModule(lr=args.lr, use_lpips=True, save_images_dir=args.save_images_dir)

    logger = TensorBoardLogger(save_dir="logs", name="lowlight")
    ckpt_psnr = ModelCheckpoint(dirpath=args.ckpt_dir, filename="best-psnr", monitor="val/psnr", mode="max", save_top_k=1)
    ckpt_ssim = ModelCheckpoint(dirpath=args.ckpt_dir, filename="best-ssim", monitor="val/ssim", mode="max", save_top_k=1)
    ckpt_lpips = ModelCheckpoint(dirpath=args.ckpt_dir, filename="best-lpips", monitor="val/lpips", mode="min", save_top_k=1)
    ckpt_last = ModelCheckpoint(dirpath=args.ckpt_dir, filename="last", save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=args.gpus, precision=args.precision, strategy="ddp_find_unused_parameters_true" if args.gpus > 1 else "auto", logger=logger, callbacks=[ckpt_psnr, ckpt_ssim, ckpt_lpips, ckpt_last, lr_monitor], log_every_n_steps=50, check_val_every_n_epoch=1)

    trainer.fit(model, train_loader, val_loader)

    print("\nTraining complete. Best model and metrics:")
    print(f" - PSNR highest: {ckpt_psnr.best_model_path}")
    print(f" - SSIM highest: {ckpt_ssim.best_model_path}")
    print(f" - LPIPS lowest: {ckpt_lpips.best_model_path}")
    print(f" - Last model: {ckpt_last.last_model_path}")

    best_epoch = getattr(model, "best_epoch", -1)
    print(f"\n[Best globally - Model statistics]")
    print(f" * Best PSNR = {model.best_psnr:.4f} dB")
    print(f" * Best SSIM = {model.best_ssim:.6f}")
    if model.use_lpips:
        print(f" * Best LPIPS = {model.best_lpips:.6f}")
    print(f" * Occurred at epoch = {best_epoch}")


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    main()
