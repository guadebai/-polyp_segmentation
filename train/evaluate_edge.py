import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_kvasir_edge import KvasirSegEdgeDataset
from train_common import (
    BATCH_SIZE,
    DATA_ROOT,
    EDGE_WIDTH,
    IMAGE_SIZE,
    NUM_WORKERS,
    THRESHOLD,
    UnetBoundary,
    get_data_split,
    make_transforms,
    set_seed,
)


MODEL_CONFIGS = {
    "lambda_0_1": Path(
        "./runs/kvasir_unet_boundary_lambda_0.1/best.pth"
    ),
    "lambda_0_3": Path(
        "./runs/kvasir_unet_boundary_lambda_0.3/best.pth"
    ),
    "lambda_0_5": Path(
        "./runs/kvasir_unet_boundary_lambda_0.5/best.pth"
    ),
    "lambda_1": Path(
        "./runs/kvasir_unet_boundary_lambda_1/best.pth"
    ),
    "lambda_0_1_pos_10": Path(
        "./runs/kvasir_unet_boundary_lambda_0.1_pos_10/best.pth"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the edge head of boundary models."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lambda_0_1"],
        help="Example: --models lambda_0_1 lambda_0_3",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
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
    )
    return parser.parse_args()


def choose_names(split):
    _, val_names, test_names = get_data_split(DATA_ROOT)

    if split == "val":
        return val_names
    return test_names


def build_loader(names, batch_size):
    dataset = KvasirSegEdgeDataset(
        root_dir=DATA_ROOT,
        file_names=names,
        image_size=IMAGE_SIZE,
        transform=make_transforms(False),
        edge_width=EDGE_WIDTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def load_checkpoint(model, checkpoint, device):
    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    try:
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            checkpoint,
            map_location=device,
        )

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
    fp = int(
        np.logical_and(
            pred,
            np.logical_not(gt),
        ).sum()
    )
    fn = int(
        np.logical_and(
            np.logical_not(pred),
            gt,
        ).sum()
    )
    tn = int(
        np.logical_and(
            np.logical_not(pred),
            np.logical_not(gt),
        ).sum()
    )

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    dice = (
        1.0
        if dice_den == 0
        else (2.0 * tp) / dice_den
    )
    iou = (
        1.0
        if iou_den == 0
        else tp / iou_den
    )

    precision = (
        1.0
        if precision_den == 0 and recall_den == 0
        else 0.0
        if precision_den == 0
        else tp / precision_den
    )

    recall = (
        1.0
        if recall_den == 0
        else tp / recall_den
    )

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


def evaluate_model(
    model_name,
    checkpoint,
    loader,
    dataset_size,
    threshold,
    device,
):
    model = UnetBoundary(
        encoder_weights=None
    ).to(device).float()

    load_checkpoint(
        model,
        checkpoint,
        device,
    )
    model.eval()

    dice_values = []
    iou_values = []
    precision_values = []
    recall_values = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"Evaluate edge {model_name}",
        ):
            images, _, edge_gt, _ = batch

            images = images.to(device).float()
            edge_gt = edge_gt.to(device).float()

            _, logits_edge = model(images)
            edge_probs = torch.sigmoid(logits_edge)

            edge_preds = (
                edge_probs >= threshold
            ).cpu().numpy().astype(np.uint8)

            edge_gts = (
                edge_gt >= 0.5
            ).cpu().numpy().astype(np.uint8)

            for index in range(edge_preds.shape[0]):
                pred01 = edge_preds[index, 0]
                gt01 = edge_gts[index, 0]

                metrics = metric_one_image(
                    pred01,
                    gt01,
                )

                dice_values.append(metrics["dice"])
                iou_values.append(metrics["iou"])
                precision_values.append(
                    metrics["precision"]
                )
                recall_values.append(
                    metrics["recall"]
                )

                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]
                total_tn += metrics["tn"]

    if len(dice_values) != dataset_size:
        raise RuntimeError(
            f"Evaluated {len(dice_values)} images, "
            f"but dataset contains {dataset_size}."
        )

    return {
        "Model": model_name,
        "Checkpoint": str(checkpoint),
        "Images": dataset_size,
        "Threshold": threshold,
        "Edge_Dice": float(np.mean(dice_values)),
        "Edge_IoU": float(np.mean(iou_values)),
        "Edge_Precision": float(
            np.mean(precision_values)
        ),
        "Edge_Recall": float(
            np.mean(recall_values)
        ),
        "Edge_Total_FP": total_fp,
        "Edge_Total_TP": total_tp,
        "Edge_Total_FN": total_fn,
        "Edge_Total_TN": total_tn,
    }


def print_results(results, split):
    print("\n" + "=" * 94)
    print(f"Edge evaluation split: {split}")
    print(
        f"{'Model':<14}"
        f"{'Edge Dice':>12}"
        f"{'Edge IoU':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'Total FP':>14}"
    )
    print("-" * 94)

    for row in results:
        print(
            f"{row['Model']:<14}"
            f"{row['Edge_Dice']:>12.4f}"
            f"{row['Edge_IoU']:>12.4f}"
            f"{row['Edge_Precision']:>12.4f}"
            f"{row['Edge_Recall']:>12.4f}"
            f"{row['Edge_Total_FP']:>14d}"
        )

    print("=" * 94)


def save_csv(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(results[0].keys())

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

    unknown = [
        name
        for name in args.models
        if name not in MODEL_CONFIGS
    ]
    if unknown:
        raise ValueError(
            f"Unknown model names: {unknown}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    names = choose_names(args.split)
    dataset, loader = build_loader(
        names,
        args.batch_size,
    )

    print(
        f"Split: {args.split}, "
        f"images: {len(dataset)}, "
        f"edge width: {EDGE_WIDTH}, "
        f"threshold: {args.threshold}"
    )

    results = []

    for model_name in args.models:
        result = evaluate_model(
            model_name=model_name,
            checkpoint=MODEL_CONFIGS[model_name],
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
        else f"./runs/evaluation_edge_{args.split}.csv"
    )
    save_csv(
        results,
        output_path,
    )


if __name__ == "__main__":
    main()