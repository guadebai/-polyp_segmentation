from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PRAP_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRAP_DIR.parent
TRAIN_DIR = REPO_ROOT / "train"

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}\n"
        "Expected repository layout: repo/train and repo/PRAP."
    )

sys.path.insert(0, str(TRAIN_DIR))

from dataset_kvasir import KvasirSegDataset  # noqa: E402
from train_common import (  # noqa: E402
    BATCH_SIZE,
    DATA_ROOT,
    IMAGE_SIZE,
    NUM_WORKERS,
    UnetBaseline,
    calculate_hd95,
    get_data_split,
    make_transforms,
    set_seed,
)

from postprocess import (  # noqa: E402
    PostprocessParams,
    apply_postprocess,
    no_postprocess,
)


SCRIPT_VERSION = "2.1-seeded"
METHODS = ("none", "area", "entropy", "seeded", "prap")
DEFAULT_CHECKPOINT = (
    TRAIN_DIR
    / "runs"
    / "kvasir_unet_baseline"
    / "best.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate None, Area, Entropy, Seeded Component, and PRAP "
            "on the same frozen Baseline probability maps."
        )
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help=(
            "Use val while selecting post-processing parameters. "
            "Use test only after all parameters are locked."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["all", *METHODS],
        default=["all"],
        help=(
            "Methods to evaluate: none, area, entropy, seeded, prap. "
            "Use 'all' to evaluate all five methods."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.45,
        help=(
            "Mean binary entropy threshold using natural logarithms. "
            "Maximum entropy is ln(2), approximately 0.6931."
        ),
    )
    parser.add_argument(
        "--seed-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--dilation-iterations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PRAP_DIR / "results",
    )
    return parser.parse_args()


def choose_names(split: str) -> list[str]:
    _, val_names, test_names = get_data_split(DATA_ROOT)

    if split == "val":
        return val_names
    if split == "test":
        return test_names

    raise ValueError(f"Unsupported split: {split}")


def build_loader(
    names: list[str],
    batch_size: int,
) -> tuple[KvasirSegDataset, DataLoader]:
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


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: Path,
    device: torch.device,
) -> None:
    checkpoint = Path(checkpoint)

    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

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

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    if not isinstance(state, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a state-dict or a "
            "dictionary containing state_dict/model_state_dict."
        )

    cleaned_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned_state, strict=True)


def metric_one_image(
    pred01: np.ndarray,
    gt01: np.ndarray,
) -> dict[str, float | int]:
    pred = np.asarray(pred01).astype(bool)
    gt = np.asarray(gt01).astype(bool)

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


def initialize_store(
    selected_methods: list[str],
) -> dict[str, dict[str, list[float] | int]]:
    store = {}

    for method in selected_methods:
        store[method] = {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "hd95": [],
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_tn": 0,
            "total_pred_pixels": 0,
            "total_removed_pixels": 0,
        }

    return store


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    names: list[str],
    selected_methods: list[str],
    params: PostprocessParams,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    aggregate_store = initialize_store(selected_methods)
    per_image_rows: list[dict] = []
    sample_index = 0

    model.eval()

    with torch.inference_mode():
        for images, masks in tqdm(
            loader,
            desc="Evaluate post-processing",
        ):
            images = images.to(device).float()
            masks = masks.to(device).float()

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            batch_probs = probabilities.cpu().numpy()
            batch_gts = (
                masks >= 0.5
            ).cpu().numpy().astype(np.uint8)

            for batch_index in range(batch_probs.shape[0]):
                if sample_index >= len(names):
                    raise RuntimeError(
                        "DataLoader produced more samples than the split list."
                    )

                sample_name = names[sample_index]
                prob = batch_probs[batch_index, 0]
                gt01 = batch_gts[batch_index, 0]

                raw_pred = no_postprocess(
                    prob,
                    candidate_threshold=params.candidate_threshold,
                )
                raw_pixels = int(raw_pred.sum())

                for method in selected_methods:
                    pred01 = apply_postprocess(
                        method,
                        prob,
                        params,
                    )
                    metrics = metric_one_image(pred01, gt01)
                    hd95 = float(calculate_hd95(pred01, gt01))
                    pred_pixels = int(pred01.sum())
                    removed_pixels = raw_pixels - pred_pixels

                    method_store = aggregate_store[method]
                    method_store["dice"].append(metrics["dice"])
                    method_store["iou"].append(metrics["iou"])
                    method_store["precision"].append(
                        metrics["precision"]
                    )
                    method_store["recall"].append(metrics["recall"])
                    method_store["hd95"].append(hd95)
                    method_store["total_tp"] += metrics["tp"]
                    method_store["total_fp"] += metrics["fp"]
                    method_store["total_fn"] += metrics["fn"]
                    method_store["total_tn"] += metrics["tn"]
                    method_store["total_pred_pixels"] += pred_pixels
                    method_store["total_removed_pixels"] += (
                        removed_pixels
                    )

                    per_image_rows.append(
                        {
                            "Sample": sample_name,
                            "Method": method,
                            "Dice": metrics["dice"],
                            "IoU": metrics["iou"],
                            "Precision": metrics["precision"],
                            "Recall": metrics["recall"],
                            "HD95": hd95,
                            "TP": metrics["tp"],
                            "FP": metrics["fp"],
                            "FN": metrics["fn"],
                            "TN": metrics["tn"],
                            "GT_Pixels": int(gt01.sum()),
                            "Raw_Pred_Pixels": raw_pixels,
                            "Post_Pred_Pixels": pred_pixels,
                            "Removed_Pixels_vs_None": removed_pixels,
                        }
                    )

                sample_index += 1

    if sample_index != len(names):
        raise RuntimeError(
            f"Evaluated {sample_index} images, expected {len(names)}."
        )

    aggregate_rows: list[dict] = []

    for method in selected_methods:
        values = aggregate_store[method]

        aggregate_rows.append(
            {
                "Method": method,
                "Images": len(names),
                "Candidate_Threshold": params.candidate_threshold,
                "Min_Area": params.min_area,
                "Entropy_Threshold": (
                    params.entropy_threshold
                    if method == "entropy"
                    else ""
                ),
                "Seed_Threshold": (
                    params.seed_threshold
                    if method in {"seeded", "prap"}
                    else ""
                ),
                "Kernel_Size": (
                    params.kernel_size
                    if method == "prap"
                    else ""
                ),
                "Dilation_Iterations": (
                    params.dilation_iterations
                    if method == "prap"
                    else ""
                ),
                "Dice": float(np.mean(values["dice"])),
                "IoU": float(np.mean(values["iou"])),
                "Precision": float(
                    np.mean(values["precision"])
                ),
                "Recall": float(np.mean(values["recall"])),
                "HD95": float(np.mean(values["hd95"])),
                "Total_FP": int(values["total_fp"]),
                "Total_TP": int(values["total_tp"]),
                "Total_FN": int(values["total_fn"]),
                "Total_TN": int(values["total_tn"]),
                "Total_Pred_Pixels": int(
                    values["total_pred_pixels"]
                ),
                "Removed_Pixels_vs_None": int(
                    values["total_removed_pixels"]
                ),
            }
        )

    add_deltas_against_none(aggregate_rows)
    return aggregate_rows, per_image_rows


def add_deltas_against_none(
    aggregate_rows: list[dict],
) -> None:
    none_row = next(
        (
            row
            for row in aggregate_rows
            if row["Method"] == "none"
        ),
        None,
    )

    for row in aggregate_rows:
        if none_row is None:
            row["Delta_Dice_vs_None"] = ""
            row["Delta_Precision_vs_None"] = ""
            row["Delta_Recall_vs_None"] = ""
            row["FP_Reduction_vs_None_Pct"] = ""
            continue

        row["Delta_Dice_vs_None"] = (
            row["Dice"] - none_row["Dice"]
        )
        row["Delta_Precision_vs_None"] = (
            row["Precision"] - none_row["Precision"]
        )
        row["Delta_Recall_vs_None"] = (
            row["Recall"] - none_row["Recall"]
        )

        none_fp = int(none_row["Total_FP"])
        if none_fp == 0:
            row["FP_Reduction_vs_None_Pct"] = 0.0
        else:
            row["FP_Reduction_vs_None_Pct"] = (
                100.0
                * (none_fp - int(row["Total_FP"]))
                / none_fp
            )


def print_results(
    aggregate_rows: list[dict],
    split: str,
) -> None:
    print("\n" + "=" * 120)
    print(f"Post-processing evaluation split: {split}")
    print(
        f"{'Method':<10}"
        f"{'Dice':>10}"
        f"{'IoU':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'HD95':>10}"
        f"{'Total FP':>14}"
        f"{'FP red.%':>12}"
    )
    print("-" * 120)

    for row in aggregate_rows:
        fp_reduction = row["FP_Reduction_vs_None_Pct"]
        fp_reduction_text = (
            ""
            if fp_reduction == ""
            else f"{fp_reduction:.2f}"
        )

        print(
            f"{row['Method']:<10}"
            f"{row['Dice']:>10.4f}"
            f"{row['IoU']:>10.4f}"
            f"{row['Precision']:>12.4f}"
            f"{row['Recall']:>10.4f}"
            f"{row['HD95']:>10.4f}"
            f"{row['Total_FP']:>14d}"
            f"{fp_reduction_text:>12}"
        )

    print("=" * 120)


def write_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to: {output_path}")


def main() -> None:
    args = parse_args()
    set_seed()

    print(f"evaluate_postprocess.py version: {SCRIPT_VERSION}")

    selected_methods = (
        list(METHODS)
        if args.methods == ["all"]
        else list(dict.fromkeys(args.methods))
    )

    params = PostprocessParams(
        candidate_threshold=args.candidate_threshold,
        min_area=args.min_area,
        entropy_threshold=args.entropy_threshold,
        seed_threshold=args.seed_threshold,
        kernel_size=args.kernel_size,
        dilation_iterations=args.dilation_iterations,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    names = choose_names(args.split)
    dataset, loader = build_loader(
        names,
        args.batch_size,
    )

    print(
        f"Split: {args.split}, images: {len(dataset)}\n"
        f"Methods: {selected_methods}\n"
        f"Parameters: {params}"
    )

    model = UnetBaseline(
        encoder_weights=None,
    ).to(device).float()

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    aggregate_rows, per_image_rows = evaluate(
        model=model,
        loader=loader,
        names=names,
        selected_methods=selected_methods,
        params=params,
        device=device,
    )

    print_results(
        aggregate_rows,
        args.split,
    )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    aggregate_path = (
        output_dir
        / f"postprocess_{args.split}_aggregate.csv"
    )
    per_image_path = (
        output_dir
        / f"postprocess_{args.split}_per_image.csv"
    )

    write_csv(
        aggregate_rows,
        aggregate_path,
    )
    write_csv(
        per_image_rows,
        per_image_path,
    )


if __name__ == "__main__":
    main()