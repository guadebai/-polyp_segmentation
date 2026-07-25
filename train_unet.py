import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from medpy.metric.binary import hd95

from dataset_kvasir import KvasirSegDataset, list_stems
from losses import bce_dice_loss, dice_coeff

# 1 Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

# 2 Transforms
def make_transforms(train=True, image_size=352):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.10, rotate_limit=10, p=0.3, border_mode=0),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ])
    return A.Compose([])

# 3 Main
def main():
    set_seed(42)
    torch.set_default_dtype(torch.float32)

    data_root = r"D:\data set\Kvasir\Kvasir-SEG" 
    image_size = 352
    batch_size = 4 
    epochs = 40
    lr = 1e-4
    num_workers = 0 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thr_eval = 0.5 # 评估阈值

    names = sorted(list_stems(data_root))
    random.shuffle(names)
    split = int(0.8 * len(names))
    train_names, val_names = names[:split], names[split:]

    train_ds = KvasirSegDataset(data_root, train_names, image_size, make_transforms(True, image_size))
    val_ds = KvasirSegDataset(data_root, val_names, image_size, make_transforms(False, image_size))

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # UNet Baseline (No Boundary Head)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device).float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    out_dir = Path("./runs/kvasir_unet_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    patience = 8
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss_total = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for images, masks in pbar_train:
            images, masks = images.to(device).float(), masks.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(images)
                loss = bce_dice_loss(logits, masks, bce_weight=0.5)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item() * images.size(0)

        avg_train_loss = train_loss_total / len(train_ds)

        # ---- val ----
        model.eval()
        val_dice_total, val_hd95_total = 0.0, 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]")
        with torch.no_grad():
            for images, masks in pbar_val:
                images, masks = images.to(device).float(), masks.to(device).float()
                logits = model(images)
                
                probs = torch.sigmoid(logits)
                val_dice_total += dice_coeff(probs, masks).item() * images.size(0)

                # HD95 计算
                preds = (probs > thr_eval).cpu().numpy().astype(np.uint8)
                gts = masks.cpu().numpy().astype(np.uint8)
                for i in range(preds.shape[0]):
                    if np.any(preds[i]) and np.any(gts[i]):
                        val_hd95_total += hd95(preds[i, 0], gts[i, 0])
                    else:
                        val_hd95_total += 50.0

        avg_val_dice = val_dice_total / len(val_ds)
        avg_val_hd95 = val_hd95_total / len(val_ds)

        # 打印您要求的格式
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_dice={avg_val_dice:.4f}  val_hd95={avg_val_hd95:.4f}")

        # save best
        if avg_val_dice > best_val_dice + 1e-4:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), out_dir / "best.pth")
            no_improve = 0
        else:
            no_improve += 1
            print(f"(no improve: {no_improve}/{patience})")

        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val_dice={best_val_dice:.4f}")
            break

    print(f"Done. Best val_dice = {best_val_dice}")

if __name__ == "__main__":
    main()