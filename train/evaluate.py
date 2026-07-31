import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_kvasir import KvasirSegDataset

from train_common import (
    DATA_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    THRESHOLD,
    UnetBaseline,
    UnetBoundary,
    calculate_hd95,
    get_data_split,
    make_transforms,
    set_seed,
)

MODEL_CONFIGS = {
    "baseline": {
        "model_type": "baseline",
        "checkpoint": Path("./runs/kvasir_unet_baseline/best.pth"),
    },
    "lambda_0_1": {
        "model_type": "boundary",
        "checkpoint": Path("./runs/kvasir_unet_boundary_lambda_0.1/best.pth"),
    },
    "lambda_0_3": {
        "model_type": "boundary",
        "checkpoint": Path("./runs/kvasir_unet_boundary_lambda_0.3/best.pth"),
    },
    "lambda_0_5": {
        "model_type": "boundary",
        "checkpoint": Path("./runs/kvasir_unet_boundary_lambda_0.5/best.pth"),
    },
    "lambda_1": {
        "model_type": "boundary",
        "checkpoint": Path("./runs/kvasir_unet_boundary_lambda_1/best.pth"),
    },
     "lambda_0_1_pos_10": {
        "model_type": "boundary",
        "checkpoint": Path(
            "./runs/kvasir_unet_boundary_lambda_0.1_pos_10/best.pth"
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for Kvasir-SEG models."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="val for evaluation, test for showing the final results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "model to evaluate, for example: baseline lambda_0_3"
            "default is all, evaluate all models in MODEL_CONFIGS."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="default to THRESHOLD。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path, default to runs/evaluation_<split>.csv。",
    )
    return parser.parse_args()


def choose_names(split):
    train_names, val_names, test_names = get_data_split(DATA_ROOT)

    if split == "val":
        return val_names
    if split == "test":
        return test_names

    raise ValueError(f"Unsupported split: {split}")


def build_loader(names, batch_size):
    dataset = KvasirSegDataset(
        root_dir=DATA_ROOT,
        file_names=names,
        image_size=IMAGE_SIZE,
        transform=make_transforms(False),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def build_model(model_type, device):
    if model_type == "baseline":
        model = UnetBaseline(encoder_weights=None)
    elif model_type == "boundary":
        model = UnetBoundary(encoder_weights=None)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device).float()


def load_checkpoint(model, checkpoint, device):
    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    try:
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }

    model.load_state_dict(state, strict=True)


def metric_one_image(pred01, gt01):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    dice = 1.0 if dice_den == 0 else (2.0 * tp) / dice_den
    iou = 1.0 if iou_den == 0 else tp / iou_den

    precision = (
        1.0 if precision_den == 0 and recall_den == 0
        else 0.0 if precision_den == 0
        else tp / precision_den
    )
    recall = 1.0 if recall_den == 0 else tp / recall_den

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_one_model(
    model_name,
    config,
    loader,
    dataset_size,
    threshold,
    device,
):
    model = build_model(config["model_type"], device)
    load_checkpoint(model, config["checkpoint"], device)
    model.eval()

    dice_values = []
    iou_values = []
    precision_values = []
    recall_values = []
    hd95_values = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluate {model_name}"):
            # KvasirSegDataset当前训练版返回(images, masks)
            images, masks = batch

            images = images.to(device).float()
            masks = masks.to(device).float()

            outputs = model(images)

            if config["model_type"] == "boundary":
                logits_mask, _ = outputs
            else:
                logits_mask = outputs

            probs = torch.sigmoid(logits_mask)

            preds = (
                probs >= threshold
            ).cpu().numpy().astype(np.uint8)

            gts = (
                masks >= 0.5
            ).cpu().numpy().astype(np.uint8)

            for index in range(preds.shape[0]):
                pred01 = preds[index, 0]
                gt01 = gts[index, 0]

                metrics = metric_one_image(pred01, gt01)

                dice_values.append(metrics["dice"])
                iou_values.append(metrics["iou"])
                precision_values.append(metrics["precision"])
                recall_values.append(metrics["recall"])
                hd95_values.append(calculate_hd95(pred01, gt01))

                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]
                total_tn += metrics["tn"]

    if len(dice_values) != dataset_size:
        raise RuntimeError(
            f"Evaluated {len(dice_values)} images, "
            f"but dataset contains {dataset_size}."
        )

    result = {
        "Model": model_name,
        "Checkpoint": str(config["checkpoint"]),
        "Images": dataset_size,
        "Threshold": threshold,
        "Dice": float(np.mean(dice_values)),
        "IoU": float(np.mean(iou_values)),
        "Precision": float(np.mean(precision_values)),
        "Recall": float(np.mean(recall_values)),
        "HD95": float(np.mean(hd95_values)),
        "Total_FP": total_fp,
        "Total_TP": total_tp,
        "Total_FN": total_fn,
        "Total_TN": total_tn,
    }

    return result


def print_results(results, split):
    print("\n" + "=" * 102)
    print(f"Evaluation split: {split}")
    print(
        f"{'Model':<14}"
        f"{'Dice':>10}"
        f"{'IoU':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'HD95':>10}"
        f"{'Total FP':>14}"
    )
    print("-" * 102)

    for row in results:
        print(
            f"{row['Model']:<14}"
            f"{row['Dice']:>10.4f}"
            f"{row['IoU']:>10.4f}"
            f"{row['Precision']:>12.4f}"
            f"{row['Recall']:>10.4f}"
            f"{row['HD95']:>10.4f}"
            f"{row['Total_FP']:>14d}"
        )

    print("=" * 102)


def save_csv(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Model",
        "Checkpoint",
        "Images",
        "Threshold",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "HD95",
        "Total_FP",
        "Total_TP",
        "Total_FN",
        "Total_TN",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV saved to: {output_path}")


def main():
    args = parse_args()
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    names = choose_names(args.split)
    dataset, loader = build_loader(names, args.batch_size)

    print(
        f"Split: {args.split}, images: {len(dataset)}, "
        f"threshold: {args.threshold}"
    )

    if args.models == ["all"]:
        selected_models = list(MODEL_CONFIGS.keys())
    else:
        selected_models = args.models

    unknown = [
        name for name in selected_models
        if name not in MODEL_CONFIGS
    ]
    if unknown:
        raise ValueError(
            f"Unknown model names: {unknown}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    results = []

    for model_name in selected_models:
        config = MODEL_CONFIGS[model_name]
        result = evaluate_one_model(
            model_name=model_name,
            config=config,
            loader=loader,
            dataset_size=len(dataset),
            threshold=args.threshold,
            device=device,
        )
        results.append(result)

    print_results(results, args.split)

    output_path = (
        args.output
        if args.output is not None
        else f"./runs/evaluation_{args.split}.csv"
    )
    save_csv(results, output_path)


if __name__ == "__main__":
    main()