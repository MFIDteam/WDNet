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

from net.model import PromptIR
from utils.val_utils import compute_psnr_ssim  # 你工程里已有的PSNR/SSIM
from torchvision.transforms import functional as TF
from utils.swt_loss_simple import SWTLoss

import lpips


# ---------------------------
# 工具函数
# ---------------------------
def list_images(folder: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files


def pair_by_name(src_paths: List[str], gt_root: str) -> List[Tuple[str, str]]:
    """文件名配对：优先 rain-xxx -> norain-xxx；否则同名；否则 rain 前缀 -> norain 前缀。"""
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
    # [0,1] float32 CHW
    return TF.to_tensor(img)


def crop_to_multiple_of(t: torch.Tensor, base: int = 16) -> torch.Tensor:
    """裁到 base 的倍数（等价原来 crop_img(..., base=16)）。"""
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
    """随机切 patch（输入均已 base=16 对齐）。"""
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
    if not os.path.isdir(path):
        raise RuntimeError(f"目录不存在：{path}")
    if must_contain:
        missing = [d for d in must_contain if not os.path.isdir(os.path.join(path, d))]
        if missing:
            subs = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
            raise RuntimeError(
                f"目录 {path} 缺少子目录 {missing}\n当前子目录有：{subs}"
            )


# ---------------------------
# 数据集（默认重复扩增=120）
# ---------------------------
class LowLightTrainDataset(Dataset):
    def __init__(self, train_root: str, patch_size: int = 128,
                 repeat_factor: int = 120):
        """
        train_root 形如 .../data/Train
        其下需包含 rainy/ 与 gt/
        repeat_factor：数据重复倍数（默认 120，等价你原来的 rs_ids * 120）
        """
        super().__init__()
        ensure_dir_exists(train_root, ["rainy", "gt"])
        self.patch_size = patch_size

        input_dir = os.path.join(train_root, "rainy")
        gt_dir    = os.path.join(train_root, "gt")

        srcs = list_images(input_dir)
        self.base_pairs = pair_by_name(srcs, gt_dir)
        if len(self.base_pairs) == 0:
            raise RuntimeError(f"未在 {input_dir} 与 {gt_dir} 中配到任何训练样本。")

        # —— 重复扩增（关键！默认 120）——
        self.repeat_factor = max(1, int(repeat_factor))
        self.pairs = self.base_pairs * self.repeat_factor

        print(f"[DATASET][TRAIN] 根目录: {train_root}")
        print(f"[DATASET][TRAIN] 输入: {input_dir}")
        print(f"[DATASET][TRAIN] 标签: {gt_dir}")
        print(f"[DATASET][TRAIN] 原始成对样本数: {len(self.base_pairs)} | repeat_factor={self.repeat_factor} => 训练样本数: {len(self.pairs)}")
        print("[DATASET][TRAIN] 前3个样本:")
        for sp, gp in self.base_pairs[:3]:
            print("   ", os.path.basename(sp), " <-> ", os.path.basename(gp))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, gt_path = self.pairs[idx]
        inp = Image.open(src_path).convert('RGB')
        gt  = Image.open(gt_path).convert('RGB')

        t_inp, t_gt = to_tensor_uint8(inp), to_tensor_uint8(gt)

        # base=16 对齐（等价你原来的 crop_img）
        t_inp = crop_to_multiple_of(t_inp, base=16)
        t_gt  = crop_to_multiple_of(t_gt,  base=16)

        # 随机增广：水平/垂直翻转 + 0/90/180/270 旋转（等价 random_augmentation 的集合）
        if np.random.rand() < 0.5:
            t_inp = TF.hflip(t_inp); t_gt = TF.hflip(t_gt)
        if np.random.rand() < 0.5:
            t_inp = TF.vflip(t_inp); t_gt = TF.vflip(t_gt)
        k = np.random.randint(0, 4)
        if k:
            t_inp = torch.rot90(t_inp, k, [1, 2])
            t_gt  = torch.rot90(t_gt,  k, [1, 2])

        # 随机裁剪 patch
        t_inp, t_gt = rand_crop_pair(t_inp, t_gt, self.patch_size)

        return os.path.basename(src_path), t_inp, t_gt


class LowLightValDataset(Dataset):
    def __init__(self, test_root: str):
        """
        test_root 形如 .../data/Test
        其下需包含 testA/ 与 testB/
        """
        super().__init__()
        ensure_dir_exists(test_root, ["testA", "testB"])

        input_dir = os.path.join(test_root, "testA")
        gt_dir    = os.path.join(test_root, "testB")

        srcs = list_images(input_dir)
        self.pairs = pair_by_name(srcs, gt_dir)
        if len(self.pairs) == 0:
            raise RuntimeError(f"未在 {input_dir} 与 {gt_dir} 中配到任何验证样本。")

        print(f"[DATASET][VAL] 根目录: {test_root}")
        print(f"[DATASET][VAL] 输入: {input_dir}")
        print(f"[DATASET][VAL] 标签: {gt_dir}")
        print(f"[DATASET][VAL] 成对样本数: {len(self.pairs)}")
        print("[DATASET][VAL] 前3个样本:")
        for sp, gp in self.pairs[:3]:
            print("   ", os.path.basename(sp), " <-> ", os.path.basename(gp))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, gt_path = self.pairs[idx]
        inp = Image.open(src_path).convert('RGB')
        gt  = Image.open(gt_path).convert('RGB')
        t_inp, t_gt = to_tensor_uint8(inp), to_tensor_uint8(gt)

        # 验证：不裁 patch，直接 base=16 对齐
        t_inp = crop_to_multiple_of(t_inp, base=16)
        t_gt  = crop_to_multiple_of(t_gt,  base=16)

        name = os.path.splitext(os.path.basename(src_path))[0]
        return name, t_inp, t_gt


# ---------------------------
# Lightning 模型
# ---------------------------
# ===== 基础损失组件 =====
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GradientLoss(nn.Module):
    """Sobel 梯度一致性：边缘更清晰（对 PSNR/SSIM 友好）。"""
    def __init__(self, reduction='mean'):
        super().__init__()
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,  2,  1],
                           [0,  0,  0],
                           [-1, -2, -1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.reduction = reduction

    def _grad(self, x):
        # 对每个通道做 depthwise conv
        B, C, H, W = x.shape
        kx = self.kx.repeat(C, 1, 1, 1)
        ky = self.ky.repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return gx, gy

    def forward(self, pred, target):
        pgx, pgy = self._grad(pred)
        tgx, tgy = self._grad(target)
        loss = (pgx - tgx).abs() + (pgy - tgy).abs()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SSIMLoss(nn.Module):
    """纯 PyTorch 可微 SSIM（窗口=11，高斯，C=3），返回 1-SSIM。"""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._create_window(window_size, sigma))

    @staticmethod
    def _gaussian(ws, sigma, device):
        gauss = torch.arange(ws, device=device) - (ws - 1) / 2.0
        gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def _create_window(self, ws, sigma):
        g = self._gaussian(ws, sigma, device=torch.device('cpu')).unsqueeze(1)
        window_2d = g @ g.t()
        window_2d = window_2d / window_2d.sum()
        # 变成 (1,1,ws,ws)，训练时会按通道复制
        return window_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x, y):
        # 假设输入 [0,1]
        B, C, H, W = x.shape
        window = self.window.to(x.device, x.dtype)
        window = window.expand(C, 1, self.window_size, self.window_size)

        mu_x = F.conv2d(x, window, padding=self.window_size//2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size//2, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x  = F.conv2d(x * x, window, padding=self.window_size//2, groups=C) - mu_x2
        sigma_y  = F.conv2d(y * y, window, padding=self.window_size//2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size//2, groups=C) - mu_xy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2) + 1e-12)
        # 平均成标量；损失为 1-ssim
        return (1.0 - ssim_map.mean())


# ===== 组合损失：支持权重随训练步数动态调度 =====
class CompositeLoss(nn.Module):
    def __init__(self, lambda_charb=1.0, lambda_ssim=0.2, lambda_grad=0.05,
                 lambda_swt=0.05, lambda_lpips=0.0,
                 swt_cfg=None):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim  = SSIMLoss()
        self.grad  = GradientLoss()
        self.swt   = SWTLoss(**(swt_cfg or dict(
            loss_weight_ll=0.01, loss_weight_lh=0.01, loss_weight_hl=0.01, loss_weight_hh=0.01,
            wave='sym19', mode='periodic', J=1, reduction='mean'
        )))
        self.lambda_charb = float(lambda_charb)
        self.lambda_ssim  = float(lambda_ssim)
        self.lambda_grad  = float(lambda_grad)
        self.lambda_swt   = float(lambda_swt)
        self.lambda_lpips = float(lambda_lpips)

        # LPIPS 在外部创建（避免显存重复）
        self.lpips_metric = None

    def set_lpips(self, lpips_metric):
        self.lpips_metric = lpips_metric

    def forward(self, pred, gt, lpips_weight_override=None):
        loss_c = self.charb(pred, gt)
        loss_s = self.ssim(pred, gt)
        loss_g = self.grad(pred, gt)
        loss_w = self.swt(pred, gt)

        total = (self.lambda_charb * loss_c
                 + self.lambda_ssim * loss_s
                 + self.lambda_grad * loss_g
                 + self.lambda_swt  * loss_w)

        comp = {
            "l_charb": loss_c,
            "l_ssim":  loss_s,
            "l_grad":  loss_g,
            "l_swt":   loss_w,
            "l_lpips": torch.tensor(0.0, device=pred.device)
        }

        # 可选 LPIPS（在中后期再逐步加强，对 LPIPS 指标更友好）
        lam_lp = self.lambda_lpips if lpips_weight_override is None else float(lpips_weight_override)
        if self.lpips_metric is not None and lam_lp > 0:
            p = (pred * 2 - 1).clamp(-1, 1)
            g = (gt   * 2 - 1).clamp(-1, 1)
            if p.dim() == 3:  # (C,H,W) -> (1,C,H,W)
                p = p.unsqueeze(0); g = g.unsqueeze(0)
            l_lp = self.lpips_metric(p, g).mean()
            comp["l_lpips"] = l_lp
            total = total + lam_lp * l_lp

        return total, comp


class LowLightModule(pl.LightningModule):
    def __init__(self, lr: float = 2e-4, use_lpips: bool = True, save_images_dir: str = None,
                 # 组合损失初值（可按需微调）
                 w_charb=1.0, w_ssim=0.2, w_grad=0.05, w_swt=0.05, w_lpips_max=0.2,
                 # LPIPS 延迟与平滑开启（按 step）
                 lpips_warmup_steps=5000, lpips_ramp_steps=15000):
        super().__init__()
        self.save_hyperparameters(ignore=['save_images_dir'])
        self.net = PromptIR()

        # 组合损失
        self.crit = CompositeLoss(lambda_charb=w_charb, lambda_ssim=w_ssim,
                                  lambda_grad=w_grad,  lambda_swt=w_swt,
                                  lambda_lpips=0.0)  # 先设 0，后面动态覆盖
        self.use_lpips = bool(use_lpips)
        if self.use_lpips:
            self.lpips_metric = lpips.LPIPS(net='alex')
            self.lpips_metric.eval()
            for p in self.lpips_metric.parameters():
                p.requires_grad = False
            self.crit.set_lpips(self.lpips_metric)

        self.save_images_dir = save_images_dir

        # 验证统计
        self._val_psnr, self._val_ssim, self._val_lpips = [], [], []
        self.best_psnr = float("-inf")
        self.best_ssim = float("-inf")
        self.best_lpips = float("inf")
        self.best_epoch = -1

        # LPIPS 动态权重参数
        self.lpips_warmup_steps = int(lpips_warmup_steps)
        self.lpips_ramp_steps   = int(lpips_ramp_steps)
        self.w_lpips_max        = float(w_lpips_max)

    def setup(self, stage=None):
        if self.use_lpips:
            self.lpips_metric.to(self.device)

    # —— 余弦调度 + AdamW（与你现有保持一致） ——
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.net(x)

    # —— 动态 LPIPS 权重：warmup -> ramp 到上限 ——
    def _lpips_weight(self, step: int) -> float:
        if not self.use_lpips:
            return 0.0
        if step < self.lpips_warmup_steps:
            return 0.0
        # 平滑从 0 -> w_lpips_max
        t = min(1.0, (step - self.lpips_warmup_steps) / max(1, self.lpips_ramp_steps))
        # 用 0.5*(1-cos) 更平滑
        return self.w_lpips_max * (0.5 * (1 - math.cos(math.pi * t)))

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        pred = self.net(x)

        # 动态 LPIPS 权重（按 global_step）
        cur_w_lp = self._lpips_weight(self.global_step)
        total, comp = self.crit(pred, y, lpips_weight_override=cur_w_lp)

        # 日志
        self.log("train/total", total, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/l_charb", comp["l_charb"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train/l_ssim",  comp["l_ssim"],  on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train/l_grad",  comp["l_grad"],  on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train/l_swt",   comp["l_swt"],   on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if self.use_lpips:
            self.log("train/l_lpips", comp["l_lpips"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log("train/w_lpips", cur_w_lp,        on_step=True, on_epoch=False, logger=True, sync_dist=True)
        return total

    def _lpips_eval(self, pred, gt):
        p = (pred * 2 - 1).clamp(-1, 1)
        g = (gt   * 2 - 1).clamp(-1, 1)
        if p.dim() == 3:
            p = p.unsqueeze(0); g = g.unsqueeze(0)
        return self.lpips_metric(p, g).mean()

    def on_validation_epoch_start(self):
        self._val_psnr, self._val_ssim, self._val_lpips = [], [], []

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
            from PIL import Image
            Image.fromarray(out).save(os.path.join(self.save_images_dir, f"{_name}_pred.png"))

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        import numpy as np
        psnr_mean  = float(np.mean(self._val_psnr)) if self._val_psnr else 0.0
        ssim_mean  = float(np.mean(self._val_ssim)) if self._val_ssim else 0.0
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
# 主流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    # 根目录（按你的要求设置为默认：PromptIR-main/data/Train 与 /data/Test）
    parser.add_argument("--train_root", type=str,
                        default="/media/ahu/f8177856-d0b5-47f5-a1a9-157aefbf3b67/ZXD/LLIE/MambaIR/data/LSRW-Huawei/Nikon/Train",
                        help="训练根目录，需包含 rainy/ 与 gt/")
    parser.add_argument("--test_root", type=str,
                        default="/media/ahu/f8177856-d0b5-47f5-a1a9-157aefbf3b67/ZXD/LLIE/MambaIR/data/LSRW-Huawei/Nikon/Test",
                        help="验证/测试根目录，需包含 testA/ 与 testB/")

    # 训练超参（默认按你之前设置）
    parser.add_argument("--epochs", type=int, default=1266)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=16)

    # —— 关键：默认就用 120 倍重复扩增（恢复原始步数/epoch 的规模）——
    parser.add_argument("--repeat_factor", type=int, default=1)

    # 硬件/精度（默认双卡 + 全精度）
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--precision", type=str, default="32", choices=["16", "32"])

    # 日志与权重
    parser.add_argument("--ckpt_dir", type=str, default="train_ckpt")
    parser.add_argument("--save_images_dir", type=str, default="val_vis")

    args = parser.parse_args()

    # 随机性
    pl.seed_everything(0, workers=True)

    # 构建数据集/加载器
    train_set = LowLightTrainDataset(args.train_root, args.patch_size, repeat_factor=args.repeat_factor)
    val_set   = LowLightValDataset(args.test_root)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True
    )

    # 数据量检查
    print("\n[DATA CHECK]")
    print(f"  base_train_pairs = {len(train_set.base_pairs)}")
    print(f"  repeat_factor    = {args.repeat_factor}")
    print(f"  effective_pairs  = {len(train_set)}")
    print(f"  batch_size       = {args.batch_size}")
    print(f"  steps/epoch      = {len(train_loader)}")
    print(f"  #val pairs       = {len(val_set)}\n")

    # 模型
    model = LowLightModule(lr=args.lr, use_lpips=True, save_images_dir=args.save_images_dir)

    # 日志 & 回调（保存最佳PSNR、最佳SSIM、最佳LPIPS）
    logger = TensorBoardLogger(save_dir="logs", name="lowlight")
    ckpt_psnr = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="best-psnr",
        monitor="val/psnr",
        mode="max",
        save_top_k=1
    )
    ckpt_ssim = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="best-ssim",
        monitor="val/ssim",
        mode="max",
        save_top_k=1
    )
    ckpt_lpips = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="best-lpips",
        monitor="val/lpips",
        mode="min",
        save_top_k=1
    )
    ckpt_last = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="last",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 训练器
    strategy = "ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        precision=args.precision,
        strategy=strategy,
        logger=logger,
        callbacks=[ckpt_psnr, ckpt_ssim, ckpt_lpips, ckpt_last, lr_monitor],
        log_every_n_steps=50,
        check_val_every_n_epoch=1
    )

    # 训练 + 验证（每轮自动打印 PSNR/SSIM/LPIPS；保存最优模型）
    trainer.fit(model, train_loader, val_loader)

    print("\n训练完成。最佳模型与指标：")
    # 仍然展示各回调保存的最佳权重路径（方便你加载）
    print(f" - PSNR最高：{ckpt_psnr.best_model_path}")
    print(f" - SSIM最高：{ckpt_ssim.best_model_path}")
    print(f" - LPIPS最低：{ckpt_lpips.best_model_path}")
    print(f" - 最后一次：{ckpt_last.last_model_path}")

    # 指标值以训练过程中自己累计的全局最优为准（跨进程一致）
    best_epoch = getattr(model, "best_epoch", -1)
    print("\n[全局最优 - 由模型统计]")
    print(f" * 最佳PSNR = {model.best_psnr:.4f} dB")
    print(f" * 最佳SSIM = {model.best_ssim:.6f}")
    if model.use_lpips:
        print(f" * 最佳LPIPS = {model.best_lpips:.6f}")
    print(f" * 发生在 epoch = {best_epoch}")


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    main()
