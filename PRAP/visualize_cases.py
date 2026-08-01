from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from evaluate_postprocess import (
    DEFAULT_CHECKPOINT,
    REPO_ROOT,
    build_loader,
    choose_names,
    load_checkpoint,
    metric_one_image,
)
from postprocess import PostprocessParams, apply_postprocess
from train_common import BATCH_SIZE, UnetBaseline, set_seed


SCRIPT_VERSION = "1.1-batch-reproducible"

PRAP_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_TEST_DIR = PRAP_DIR / "results_test"
DEFAULT_OUTPUT_DIR = PRAP_DIR / "qualitative_cases"

METHOD_ORDER = ("none", "area", "entropy", "seeded", "prap")
METHOD_TITLES = {
    "none": "Baseline",
    "area": "Area",
    "entropy": "Entropy",
    "seeded": "Seeded",
    "prap": "PRAP",
}

CSV_PATHS = {
    "area": Path("area_a320/postprocess_test_per_image.csv"),
    "entropy": Path("entropy_e035/postprocess_test_per_image.csv"),
    "seeded": Path("seeded_s085/postprocess_test_per_image.csv"),
    "prap": Path("prap_s095_d5_a320/postprocess_test_per_image.csv"),
}

FLOAT_FIELDS = ("Dice", "IoU", "Precision", "Recall", "HD95")
INT_FIELDS = (
    "TP",
    "FP",
    "FN",
    "TN",
    "GT_Pixels",
    "Raw_Pred_Pixels",
    "Post_Pred_Pixels",
    "Removed_Pixels_vs_None",
)


@dataclass(frozen=True)
class SelectedCase:
    case_type: str
    rank: int
    sample: str
    fp_reduction: int
    delta_dice: float
    delta_precision: float
    delta_recall: float
    fn_increase: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically select and visualize PRAP success and failure cases "
            "from the fixed 150-image test set."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--results-test-dir",
        type=Path,
        default=DEFAULT_RESULTS_TEST_DIR,
        help="Directory containing the four formal test-result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--success-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--failure-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--neutral-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-dice-drop",
        type=float,
        default=0.005,
        help="Maximum allowed Dice decrease for automatic success cases.",
    )
    parser.add_argument(
        "--max-recall-drop",
        type=float,
        default=0.015,
        help="Maximum allowed Recall decrease for automatic success cases.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--strict-metric-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Regenerate predictions and verify Dice/Precision/Recall/FP/FN "
            "against the saved per-image CSV values."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path = REPO_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("success_count", "failure_count", "neutral_count"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.max_dice_drop < 0:
        raise ValueError("--max-dice-drop must be non-negative.")
    if args.max_recall_drop < 0:
        raise ValueError("--max-recall-drop must be non-negative.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def parse_csv_row(row: dict[str, str]) -> dict[str, str | float | int]:
    parsed: dict[str, str | float | int] = {
        "Sample": row["Sample"].strip(),
        "Method": row["Method"].strip().lower(),
    }
    for field in FLOAT_FIELDS:
        parsed[field] = float(row[field])
    for field in INT_FIELDS:
        parsed[field] = int(float(row[field]))
    return parsed


def read_method_file(
    csv_path: Path,
    expected_method: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Per-image CSV not found: {csv_path}")

    none_rows: dict[str, dict] = {}
    method_rows: dict[str, dict] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Sample", "Method", *FLOAT_FIELDS, *INT_FIELDS}
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns {sorted(missing)}: {csv_path}"
            )

        for raw_row in reader:
            row = parse_csv_row(raw_row)
            sample = str(row["Sample"])
            method = str(row["Method"])

            if method == "none":
                if sample in none_rows:
                    raise ValueError(
                        f"Duplicate none row for sample {sample}: {csv_path}"
                    )
                none_rows[sample] = row
            elif method == expected_method:
                if sample in method_rows:
                    raise ValueError(
                        f"Duplicate {expected_method} row for sample "
                        f"{sample}: {csv_path}"
                    )
                method_rows[sample] = row

    if not none_rows or not method_rows:
        raise ValueError(
            f"Expected none and {expected_method} rows in {csv_path}."
        )
    if set(none_rows) != set(method_rows):
        raise ValueError(
            f"Sample mismatch between none and {expected_method} in {csv_path}."
        )

    return none_rows, method_rows


def rows_match(left: dict, right: dict, tolerance: float = 1e-12) -> bool:
    for field in FLOAT_FIELDS:
        if abs(float(left[field]) - float(right[field])) > tolerance:
            return False
    for field in INT_FIELDS:
        if int(left[field]) != int(right[field]):
            return False
    return True


def load_formal_test_records(
    results_test_dir: Path,
) -> dict[str, dict[str, dict]]:
    records: dict[str, dict[str, dict]] = {}
    canonical_none: dict[str, dict] | None = None

    for method in ("area", "entropy", "seeded", "prap"):
        csv_path = results_test_dir / CSV_PATHS[method]
        none_rows, method_rows = read_method_file(csv_path, method)

        if canonical_none is None:
            canonical_none = none_rows
        else:
            if set(canonical_none) != set(none_rows):
                raise ValueError(
                    f"None sample set differs in {csv_path}."
                )
            mismatched = [
                sample
                for sample in canonical_none
                if not rows_match(canonical_none[sample], none_rows[sample])
            ]
            if mismatched:
                raise ValueError(
                    "The repeated none rows are inconsistent across formal "
                    f"test CSVs. First mismatch: {mismatched[0]}"
                )

        records[method] = method_rows

    if canonical_none is None:
        raise RuntimeError("No formal test records were loaded.")

    records["none"] = canonical_none

    sample_sets = [set(records[method]) for method in METHOD_ORDER]
    if any(sample_set != sample_sets[0] for sample_set in sample_sets[1:]):
        raise ValueError("Method CSVs do not contain the same test samples.")

    return records


def build_case_table(
    records: dict[str, dict[str, dict]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    for sample in sorted(records["none"]):
        none = records["none"][sample]
        prap = records["prap"][sample]

        rows.append(
            {
                "sample": sample,
                "fp_reduction": int(none["FP"]) - int(prap["FP"]),
                "delta_dice": float(prap["Dice"]) - float(none["Dice"]),
                "delta_precision": (
                    float(prap["Precision"]) - float(none["Precision"])
                ),
                "delta_recall": (
                    float(prap["Recall"]) - float(none["Recall"])
                ),
                "fn_increase": int(prap["FN"]) - int(none["FN"]),
                "removed_pixels": int(prap["Removed_Pixels_vs_None"]),
            }
        )

    return rows


def select_cases(
    table: list[dict[str, float | int | str]],
    success_count: int,
    failure_count: int,
    neutral_count: int,
    max_dice_drop: float,
    max_recall_drop: float,
) -> list[SelectedCase]:
    success_candidates = [
        row
        for row in table
        if int(row["fp_reduction"]) > 0
        and float(row["delta_dice"]) >= -max_dice_drop
        and float(row["delta_recall"]) >= -max_recall_drop
    ]
    success_candidates.sort(
        key=lambda row: (
            int(row["fp_reduction"]),
            float(row["delta_precision"]),
            float(row["delta_dice"]),
        ),
        reverse=True,
    )

    if len(success_candidates) < success_count:
        fallback = sorted(
            table,
            key=lambda row: (
                int(row["fp_reduction"]),
                float(row["delta_precision"]),
            ),
            reverse=True,
        )
        existing = {str(row["sample"]) for row in success_candidates}
        success_candidates.extend(
            row for row in fallback if str(row["sample"]) not in existing
        )

    selected_success = success_candidates[:success_count]
    used = {str(row["sample"]) for row in selected_success}

    failure_candidates = [
        row for row in table if str(row["sample"]) not in used
    ]
    failure_candidates.sort(
        key=lambda row: (
            float(row["delta_recall"]),
            float(row["delta_dice"]),
            -int(row["fn_increase"]),
        )
    )
    selected_failure = failure_candidates[:failure_count]
    used.update(str(row["sample"]) for row in selected_failure)

    neutral_candidates = [
        row for row in table if str(row["sample"]) not in used
    ]
    neutral_candidates.sort(
        key=lambda row: (
            abs(float(row["delta_dice"]))
            + abs(float(row["delta_precision"]))
            + abs(float(row["delta_recall"])),
            abs(int(row["fp_reduction"])),
        )
    )
    selected_neutral = neutral_candidates[:neutral_count]

    selected: list[SelectedCase] = []
    for case_type, rows in (
        ("success", selected_success),
        ("failure", selected_failure),
        ("neutral", selected_neutral),
    ):
        for rank, row in enumerate(rows, start=1):
            selected.append(
                SelectedCase(
                    case_type=case_type,
                    rank=rank,
                    sample=str(row["sample"]),
                    fp_reduction=int(row["fp_reduction"]),
                    delta_dice=float(row["delta_dice"]),
                    delta_precision=float(row["delta_precision"]),
                    delta_recall=float(row["delta_recall"]),
                    fn_increase=int(row["fn_increase"]),
                )
            )

    return selected


def method_parameters() -> dict[str, PostprocessParams]:
    return {
        "none": PostprocessParams(
            candidate_threshold=0.50,
            min_area=0,
            entropy_threshold=0.45,
            seed_threshold=0.95,
            kernel_size=3,
            dilation_iterations=5,
        ),
        "area": PostprocessParams(
            candidate_threshold=0.50,
            min_area=320,
            entropy_threshold=0.45,
            seed_threshold=0.95,
            kernel_size=3,
            dilation_iterations=5,
        ),
        "entropy": PostprocessParams(
            candidate_threshold=0.50,
            min_area=0,
            entropy_threshold=0.35,
            seed_threshold=0.95,
            kernel_size=3,
            dilation_iterations=5,
        ),
        "seeded": PostprocessParams(
            candidate_threshold=0.50,
            min_area=0,
            entropy_threshold=0.45,
            seed_threshold=0.85,
            kernel_size=3,
            dilation_iterations=5,
        ),
        "prap": PostprocessParams(
            candidate_threshold=0.50,
            min_area=320,
            entropy_threshold=0.45,
            seed_threshold=0.95,
            kernel_size=3,
            dilation_iterations=5,
        ),
    }


def image_tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def binary_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return (mask >= 0.5).astype(np.uint8)


def alpha_overlay(
    image: np.ndarray,
    regions: Iterable[tuple[np.ndarray, tuple[float, float, float], float]],
) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32).copy()

    for region, color, alpha in regions:
        region_bool = np.asarray(region).astype(bool)
        if not np.any(region_bool):
            continue
        color_array = np.asarray(color, dtype=np.float32)
        output[region_bool] = (
            (1.0 - alpha) * output[region_bool]
            + alpha * color_array
        )

    return np.clip(output, 0.0, 1.0)


def gt_panel(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return alpha_overlay(
        image,
        [(gt, (0.10, 0.85, 0.20), 0.45)],
    )


def confusion_panel(
    image: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    gt_bool = gt.astype(bool)
    pred_bool = prediction.astype(bool)

    tp = np.logical_and(gt_bool, pred_bool)
    fp = np.logical_and(np.logical_not(gt_bool), pred_bool)
    fn = np.logical_and(gt_bool, np.logical_not(pred_bool))

    return alpha_overlay(
        image,
        [
            (tp, (0.10, 0.85, 0.20), 0.48),
            (fp, (1.00, 0.12, 0.12), 0.68),
            (fn, (0.10, 0.42, 1.00), 0.68),
        ],
    )


def metrics_text(metrics: dict[str, float | int]) -> str:
    return (
        f"D={float(metrics['dice']):.3f}  "
        f"P={float(metrics['precision']):.3f}  "
        f"R={float(metrics['recall']):.3f}\n"
        f"FP={int(metrics['fp'])}  FN={int(metrics['fn'])}"
    )


def verify_metrics(
    sample: str,
    method: str,
    computed: dict[str, float | int],
    saved: dict,
    tolerance: float = 1e-6,
) -> None:
    comparisons = {
        "dice": "Dice",
        "precision": "Precision",
        "recall": "Recall",
    }
    for computed_field, saved_field in comparisons.items():
        if (
            abs(
                float(computed[computed_field])
                - float(saved[saved_field])
            )
            > tolerance
        ):
            raise RuntimeError(
                f"Metric mismatch for {sample}/{method}: "
                f"{computed_field} computed={computed[computed_field]}, "
                f"saved={saved[saved_field]}."
            )

    for computed_field, saved_field in (("fp", "FP"), ("fn", "FN")):
        if int(computed[computed_field]) != int(saved[saved_field]):
            raise RuntimeError(
                f"Metric mismatch for {sample}/{method}: "
                f"{computed_field} computed={computed[computed_field]}, "
                f"saved={saved[saved_field]}."
            )


def infer_selected_cases(
    model: torch.nn.Module,
    loader,
    names: list[str],
    selected: list[SelectedCase],
    records: dict[str, dict[str, dict]],
    params_by_method: dict[str, PostprocessParams],
    device: torch.device,
    strict_metric_check: bool,
) -> dict[str, dict]:
    """
    Re-run the complete fixed test loader with the same batch size used by the
    formal evaluation, then retain only the selected samples.

    This matters because GPU convolution can produce tiny floating-point
    differences when the batch shape changes. Around the 0.5 decision
    threshold, those differences may flip a few pixels. Processing the full
    loader with BATCH_SIZE=4 reproduces the formal per-image CSV protocol.
    """
    outputs: dict[str, dict] = {}
    selected_names = {case.sample for case in selected}
    sample_index = 0

    model.eval()
    with torch.inference_mode():
        for images_cpu, masks_cpu in loader:
            images_device = images_cpu.to(device).float()
            probabilities = torch.sigmoid(model(images_device))
            batch_probs = probabilities.detach().cpu().numpy()
            batch_gts = (
                masks_cpu >= 0.5
            ).numpy().astype(np.uint8)

            for batch_index in range(batch_probs.shape[0]):
                if sample_index >= len(names):
                    raise RuntimeError(
                        "DataLoader produced more samples than the split list."
                    )

                sample_name = names[sample_index]
                if sample_name in selected_names:
                    probability_np = batch_probs[batch_index, 0]
                    image_rgb = image_tensor_to_rgb(
                        images_cpu[batch_index]
                    )
                    gt = batch_gts[batch_index, 0]

                    method_outputs: dict[str, dict] = {}
                    for method in METHOD_ORDER:
                        prediction = apply_postprocess(
                            method,
                            probability_np,
                            params_by_method[method],
                        )
                        metrics = metric_one_image(prediction, gt)

                        if strict_metric_check:
                            verify_metrics(
                                sample_name,
                                method,
                                metrics,
                                records[method][sample_name],
                            )

                        method_outputs[method] = {
                            "prediction": prediction,
                            "metrics": metrics,
                        }

                    outputs[sample_name] = {
                        "image": image_rgb,
                        "gt": gt,
                        "methods": method_outputs,
                    }

                sample_index += 1

    if sample_index != len(names):
        raise RuntimeError(
            f"Evaluated {sample_index} images, expected {len(names)}."
        )

    missing = selected_names - set(outputs)
    if missing:
        raise RuntimeError(
            "Selected samples were not regenerated: "
            + ", ".join(sorted(missing))
        )

    return outputs

def draw_case_row(
    axes: np.ndarray,
    case: SelectedCase,
    output: dict,
) -> None:
    image = output["image"]
    gt = output["gt"]

    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("Image", image, None),
        ("Ground truth", gt_panel(image, gt), None),
    ]

    for method in METHOD_ORDER:
        result = output["methods"][method]
        panels.append(
            (
                METHOD_TITLES[method],
                confusion_panel(
                    image,
                    gt,
                    result["prediction"],
                ),
                metrics_text(result["metrics"]),
            )
        )

    for axis, (title, panel, subtitle) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            title if subtitle is None else f"{title}\n{subtitle}",
            fontsize=8,
            pad=4,
        )

    axes[0].set_ylabel(
        f"{case.case_type.capitalize()} {case.rank}\n{case.sample}\n"
        f"FP reduction={case.fp_reduction}\n"
        f"ΔDice={case.delta_dice:+.3f}, "
        f"ΔRecall={case.delta_recall:+.3f}",
        fontsize=8,
        rotation=0,
        labelpad=92,
        va="center",
    )


def figure_legend() -> list[Patch]:
    return [
        Patch(
            facecolor=(0.10, 0.85, 0.20),
            label="TP / GT overlap",
        ),
        Patch(
            facecolor=(1.00, 0.12, 0.12),
            label="False positive",
        ),
        Patch(
            facecolor=(0.10, 0.42, 1.00),
            label="False negative",
        ),
    ]


def save_case_group(
    cases: list[SelectedCase],
    outputs: dict[str, dict],
    output_path_png: Path,
    dpi: int,
) -> None:
    if not cases:
        return

    column_count = 2 + len(METHOD_ORDER)
    row_count = len(cases)

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(3.05 * column_count, 3.35 * row_count),
        squeeze=False,
    )

    for row_index, case in enumerate(cases):
        draw_case_row(
            axes[row_index],
            case,
            outputs[case.sample],
        )

    figure.legend(
        handles=figure_legend(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.065, 0.045, 1.0, 1.0))
    output_path_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path_png,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_individual_cases(
    selected: list[SelectedCase],
    outputs: dict[str, dict],
    output_dir: Path,
    dpi: int,
) -> None:
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for case in selected:
        figure, axes = plt.subplots(
            1,
            2 + len(METHOD_ORDER),
            figsize=(21.4, 3.75),
            squeeze=False,
        )
        draw_case_row(
            axes[0],
            case,
            outputs[case.sample],
        )
        figure.legend(
            handles=figure_legend(),
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        figure.tight_layout(rect=(0.07, 0.10, 1.0, 1.0))

        stem = f"{case.case_type}_{case.rank}_{case.sample}"
        figure.savefig(
            individual_dir / f"{stem}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def write_selected_cases_csv(
    selected: list[SelectedCase],
    records: dict[str, dict[str, dict]],
    output_path: Path,
) -> None:
    rows: list[dict] = []

    for case in selected:
        row = {
            "Case_Type": case.case_type,
            "Rank": case.rank,
            "Sample": case.sample,
            "FP_Reduction_None_minus_PRAP": case.fp_reduction,
            "Delta_Dice_PRAP_minus_None": case.delta_dice,
            "Delta_Precision_PRAP_minus_None": case.delta_precision,
            "Delta_Recall_PRAP_minus_None": case.delta_recall,
            "FN_Increase_PRAP_minus_None": case.fn_increase,
        }

        for method in METHOD_ORDER:
            saved = records[method][case.sample]
            prefix = METHOD_TITLES[method].replace(" ", "_")
            for field in ("Dice", "IoU", "Precision", "Recall", "HD95", "FP", "FN"):
                row[f"{prefix}_{field}"] = saved[field]

        rows.append(row)

    if not rows:
        raise ValueError("No cases were selected.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    selected: list[SelectedCase],
    output_path: Path,
    strict_metric_check: bool,
) -> None:
    lines = [
        f"visualize_cases.py version: {SCRIPT_VERSION}",
        "",
        "Selection basis",
        "- Split: fixed test set (150 images)",
        "- Success: largest FP reduction while Dice drop <= 0.005 "
        "and Recall drop <= 0.015",
        "- Failure: largest Recall/Dice decrease after excluding successes",
        "- Parameters are the validation-selected formal settings",
        f"- Strict regenerated-metric check: {strict_metric_check}",
        "",
        "Selected cases",
    ]

    for case in selected:
        lines.append(
            f"- {case.case_type} {case.rank}: {case.sample}; "
            f"FP reduction={case.fp_reduction}; "
            f"Delta Dice={case.delta_dice:+.6f}; "
            f"Delta Precision={case.delta_precision:+.6f}; "
            f"Delta Recall={case.delta_recall:+.6f}; "
            f"FN increase={case.fn_increase}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed()

    results_test_dir = resolve_path(args.results_test_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"visualize_cases.py version: {SCRIPT_VERSION}")
    print(f"Formal test results: {results_test_dir}")
    print(f"Output directory: {output_dir}")

    records = load_formal_test_records(results_test_dir)
    table = build_case_table(records)
    selected = select_cases(
        table=table,
        success_count=args.success_count,
        failure_count=args.failure_count,
        neutral_count=args.neutral_count,
        max_dice_drop=args.max_dice_drop,
        max_recall_drop=args.max_recall_drop,
    )

    if not selected:
        raise RuntimeError("No qualitative cases were selected.")

    print("Selected samples:")
    for case in selected:
        print(
            f"  {case.case_type:<7} {case.rank}: {case.sample} | "
            f"FP reduction={case.fp_reduction}, "
            f"Delta Dice={case.delta_dice:+.4f}, "
            f"Delta Recall={case.delta_recall:+.4f}"
        )

    names = choose_names("test")
    _, loader = build_loader(
        names,
        batch_size=BATCH_SIZE,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = UnetBaseline(
        encoder_weights=None,
    ).to(device).float()
    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("Regenerating selected predictions...")

    outputs = infer_selected_cases(
        model=model,
        loader=loader,
        names=names,
        selected=selected,
        records=records,
        params_by_method=method_parameters(),
        device=device,
        strict_metric_check=args.strict_metric_check,
    )

    grouped = {
        case_type: [
            case for case in selected if case.case_type == case_type
        ]
        for case_type in ("success", "failure", "neutral")
    }

    for case_type, cases in grouped.items():
        if not cases:
            continue
        save_case_group(
            cases=cases,
            outputs=outputs,
            output_path_png=output_dir / f"{case_type}_cases.png",
            dpi=args.dpi,
        )

    save_individual_cases(
        selected=selected,
        outputs=outputs,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    write_selected_cases_csv(
        selected=selected,
        records=records,
        output_path=output_dir / "selected_cases.csv",
    )
    write_summary(
        selected=selected,
        output_path=output_dir / "selection_summary.txt",
        strict_metric_check=args.strict_metric_check,
    )

    print("\nSaved qualitative outputs:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()