import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp
from medpy.metric.binary import hd95
from dataset_kvasir_edge import KvasirSegEdgeDataset, list_stems
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
def make_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.10, rotate_limit=10, p=0.3, border_mode=0),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ])
    return A.Compose([])

# 3) Model
class UnetMaskEdge(torch.nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.edge_head = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)

        torch.nn.init.xavier_uniform_(self.edge_head.weight, gain=1.0)
        torch.nn.init.constant_(self.edge_head.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.segmentation_head.weight, gain=1.0)
        torch.nn.init.constant_(self.segmentation_head.bias, 0.0)

    def forward(self, x):
        feats = self.encoder(x)
        dec_out = self.decoder(feats)
        logits_mask = self.segmentation_head(dec_out)
        logits_edge = self.edge_head(dec_out)
        return logits_mask, logits_edge

# 5 Main
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

    edge_width = 3
    lambda_edge = 0.5
    thr_eval = 0.5 

    stems = sorted(list_stems(data_root))
    random.shuffle(stems)
    split = int(0.8 * len(stems))
    train_names, val_names = stems[:split], stems[split:]

    train_ds = KvasirSegEdgeDataset(data_root, train_names, image_size, make_transforms(True), edge_width)
    val_ds = KvasirSegEdgeDataset(data_root, val_names, image_size, make_transforms(False), edge_width)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)

    model = UnetMaskEdge().to(device).float()
    # weight = 30
    bce_edge = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    out_dir = Path("./runs/kvasir_unet_boundary_0.5")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    best_val_hd95 = 50.0 
    patience = 8
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_total = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for images, masks, edges, _ in pbar_train:
            images, masks, edges = images.to(device).float(), masks.to(device).float(), edges.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            
            logits_mask, logits_edge = model(images)
            loss_seg = bce_dice_loss(logits_mask, masks, bce_weight=0.5)
            loss_edge = bce_edge(logits_edge, edges)
            loss = loss_seg + lambda_edge * loss_edge

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            train_loss_total += loss.item() * images.size(0)

        avg_train_loss = train_loss_total / len(train_ds)

        # Validate
        model.eval()
        val_dice_total, val_hd95_total = 0.0, 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
        with torch.no_grad():
            for images, masks, _, _ in pbar_val:
                images, masks = images.to(device).float(), masks.to(device).float()
                logits_mask, _ = model(images)
                
                probs = torch.sigmoid(logits_mask)
                val_dice_total += dice_coeff(probs, masks).item() * images.size(0)
                
                preds = (probs > thr_eval).cpu().numpy().astype(np.uint8)
                gts = masks.cpu().numpy().astype(np.uint8)
                for i in range(preds.shape[0]):
                    # only calculate HD95 when predicts and ground truth both exist
                    if np.any(preds[i]) and np.any(gts[i]):
                        val_hd95_total += hd95(preds[i, 0], gts[i, 0])
                    elif not np.any(preds[i]) and not np.any(gts[i]):
                        val_hd95_total += 0.0
                    else:
                        val_hd95_total += 50.0

        avg_val_dice = val_dice_total / len(val_ds)
        avg_val_hd95 = val_hd95_total / len(val_ds)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_dice={avg_val_dice:.4f}  val_hd95={avg_val_hd95:.4f}")

        # Early Stopping & Saving
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_val_hd95 = avg_val_hd95
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"🔥 New Best! Model saved at epoch {epoch}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"(no improve: {no_improve}/{patience})")
        
        scheduler.step()
        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print("\n" + "="*50)
    print(f"Training Finished")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Corresponding HD95: {best_val_hd95:.4f}")
    print(f"Checkpoint saved to: {out_dir / 'best.pth'}")
    print("="*50)

if __name__ == "__main__":
    main()