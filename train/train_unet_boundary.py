import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_kvasir_edge import KvasirSegEdgeDataset
from losses import bce_dice_loss, dice_coeff
from train_common import (
    DATA_ROOT,
    IMAGE_SIZE,
    EDGE_WIDTH,
    BATCH_SIZE,
    EPOCHS,
    NUM_WORKERS,
    BCE_WEIGHT,
    EDGE_POS_WEIGHT,
    LAMBDA_EDGE,
    CLIP_NORM,
    PATIENCE,
    THRESHOLD,
    BOUNDARY_OUT_DIR,
    UnetBoundary,
    calculate_hd95,
    get_data_split,
    make_optimizer,
    make_scheduler,
    make_transforms,
    set_seed,
)


def main():
    set_seed()
    torch.set_default_dtype(torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_names, val_names, _ = get_data_split(DATA_ROOT)

    train_ds = KvasirSegEdgeDataset(
        root_dir=DATA_ROOT,
        file_names=train_names,
        image_size=IMAGE_SIZE,
        transform=make_transforms(True),
        edge_width=EDGE_WIDTH
    )

    val_ds = KvasirSegEdgeDataset(
        root_dir=DATA_ROOT,
        file_names=val_names,
        image_size=IMAGE_SIZE,
        transform=make_transforms(False),
        edge_width=EDGE_WIDTH
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = UnetBoundary().to(device).float()

    bce_edge = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [EDGE_POS_WEIGHT],
            device=device
        )
    )

    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    BOUNDARY_OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_dice = 0.0
    best_val_hd95 = 50.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_loss_total = 0.0

        pbar_train = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS} [train]",
            leave=False
        )

        for images, masks, edges, _ in pbar_train:
            images = images.to(device).float()
            masks = masks.to(device).float()
            edges = edges.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            logits_mask, logits_edge = model(images)

            loss_seg = bce_dice_loss(
                logits_mask,
                masks,
                bce_weight=BCE_WEIGHT
            )

            loss_edge = bce_edge(
                logits_edge,
                edges
            )

            loss = (
                loss_seg
                + LAMBDA_EDGE * loss_edge
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=CLIP_NORM
            )

            optimizer.step()

            train_loss_total += (
                loss.item() * images.size(0)
            )

        avg_train_loss = (
            train_loss_total / len(train_ds)
        )

        # Validation
        model.eval()
        val_dice_total = 0.0
        val_hd95_total = 0.0

        pbar_val = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{EPOCHS} [val]",
            leave=False
        )

        with torch.no_grad():
            for images, masks, _, _ in pbar_val:
                images = images.to(device).float()
                masks = masks.to(device).float()

                logits_mask, _ = model(images)
                probs = torch.sigmoid(logits_mask)

                val_dice_total += (
                    dice_coeff(probs, masks).item()
                    * images.size(0)
                )

                preds = (
                    probs > THRESHOLD
                ).cpu().numpy().astype(np.uint8)

                gts = (
                    masks.cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                for i in range(preds.shape[0]):
                    val_hd95_total += calculate_hd95(
                        preds[i, 0],
                        gts[i, 0]
                    )

        avg_val_dice = (
            val_dice_total / len(val_ds)
        )
        avg_val_hd95 = (
            val_hd95_total / len(val_ds)
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f} "
            f"val_dice={avg_val_dice:.4f} "
            f"val_hd95={avg_val_hd95:.4f}"
        )

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_val_hd95 = avg_val_hd95

            torch.save(
                model.state_dict(),
                BOUNDARY_OUT_DIR / "best.pth"
            )

            print(
                f"New best model saved at epoch {epoch}"
            )
            no_improve = 0
        else:
            no_improve += 1
            print(
                f"(no improve: "
                f"{no_improve}/{PATIENCE})"
            )

        scheduler.step()

        if no_improve >= PATIENCE:
            print(
                f"Early stopping triggered "
                f"at epoch {epoch}."
            )
            break

    print("\n" + "=" * 50)
    print("Training finished")
    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
    )
    print(
        f"Corresponding HD95: "
        f"{best_val_hd95:.4f}"
    )
    print(
        f"Checkpoint: "
        f"{BOUNDARY_OUT_DIR / 'best.pth'}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()