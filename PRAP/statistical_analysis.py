from __future__ import annotations

"""Paired statistical analysis for PRAP test-set results.

Place this file in the repository's ``PRAP`` directory and run:

    python statistical_analysis.py

The script reads the locked test-set per-image CSV files, aligns samples by
``Sample``, and compares PRAP with None, Area, Entropy, and Seeded using:

- paired Wilcoxon signed-rank tests (two-sided, Pratt zero handling),
- 95% percentile bootstrap confidence intervals for mean paired improvement,
- Holm correction across all 24 tests,
- paired rank-biserial effect size.

Positive ``Mean_Improvement`` always means PRAP is better. For Dice, IoU,
Precision, and Recall, improvement = PRAP - comparator. For HD95 and FP,
improvement = comparator - PRAP.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_FILES = {
    "area": SCRIPT_DIR
    / "results_test"
    / "area_a320"
    / "postprocess_test_per_image.csv",
    "entropy": SCRIPT_DIR
    / "results_test"
    / "entropy_e035"
    / "postprocess_test_per_image.csv",
    "seeded": SCRIPT_DIR
    / "results_test"
    / "seeded_s085"
    / "postprocess_test_per_image.csv",
    "prap": SCRIPT_DIR
    / "results_test"
    / "prap_s095_d5_a320"
    / "postprocess_test_per_image.csv",
}

METRICS = ("Dice", "IoU", "Precision", "Recall", "HD95", "FP")
HIGHER_IS_BETTER = {"Dice", "IoU", "Precision", "Recall"}
COMPARATORS = ("none", "area", "entropy", "seeded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired statistical analysis for PRAP test results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "statistics_test",
        help="Directory for statistical outputs.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=10_000,
        help="Number of paired bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap confidence intervals.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Result CSV not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Sample", "Method", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"CSV {path} is missing columns: {sorted(missing)}"
        )

    duplicated = frame.duplicated(subset=["Sample", "Method"])
    if duplicated.any():
        duplicate_rows = frame.loc[duplicated, ["Sample", "Method"]]
        raise ValueError(
            f"Duplicate Sample/Method rows found in {path}:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )
    return frame


def method_rows(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = frame.loc[frame["Method"] == method].copy()
    if selected.empty:
        raise ValueError(f"Method '{method}' was not found in the CSV.")
    return selected.set_index("Sample").sort_index()


def validate_alignment(method_frames: dict[str, pd.DataFrame]) -> None:
    reference_name = next(iter(method_frames))
    reference_index = method_frames[reference_name].index

    for name, frame in method_frames.items():
        if not reference_index.equals(frame.index):
            missing = reference_index.difference(frame.index).tolist()
            extra = frame.index.difference(reference_index).tolist()
            raise ValueError(
                f"Sample mismatch for {name}. "
                f"Missing={missing[:5]}, extra={extra[:5]}"
            )


def validate_none_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    none_frames = {
        name: method_rows(frame, "none")
        for name, frame in frames.items()
    }
    validate_alignment(none_frames)

    reference_name = "prap"
    reference = none_frames[reference_name]
    for name, frame in none_frames.items():
        if name == reference_name:
            continue
        values_equal = np.allclose(
            reference.loc[:, METRICS].to_numpy(dtype=float),
            frame.loc[:, METRICS].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        if not values_equal:
            raise ValueError(
                "The duplicated None rows are inconsistent between "
                f"the PRAP CSV and the {name} CSV."
            )
    return reference


def paired_bootstrap_ci(
    values: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap input must be a non-empty 1D array.")
    if repeats < 1000:
        raise ValueError("Use at least 1000 bootstrap repeats.")

    sample_indices = rng.integers(
        0,
        values.size,
        size=(repeats, values.size),
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(lower), float(upper)


def paired_rank_biserial(improvements: np.ndarray) -> float:
    values = np.asarray(improvements, dtype=float)
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero), method="average")
    positive_sum = float(ranks[nonzero > 0].sum())
    negative_sum = float(ranks[nonzero < 0].sum())
    denominator = positive_sum + negative_sum
    if denominator == 0:
        return 0.0
    return (positive_sum - negative_sum) / denominator


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("P-values must lie in [0, 1].")

    count = p_values.size
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0

    for position, original_index in enumerate(order):
        corrected = (count - position) * p_values[original_index]
        running_max = max(running_max, corrected)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted


def wilcoxon_paired(improvements: np.ndarray) -> tuple[float, float]:
    values = np.asarray(improvements, dtype=float)
    if np.all(values == 0):
        return 0.0, 1.0

    result = wilcoxon(
        values,
        zero_method="pratt",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def build_merged_table(
    none: pd.DataFrame,
    area: pd.DataFrame,
    entropy: pd.DataFrame,
    seeded: pd.DataFrame,
    prap: pd.DataFrame,
) -> pd.DataFrame:
    method_frames = {
        "none": none,
        "area": area,
        "entropy": entropy,
        "seeded": seeded,
        "prap": prap,
    }
    validate_alignment(method_frames)

    parts = []
    for method, frame in method_frames.items():
        selected = frame.loc[:, METRICS].copy()
        selected.columns = [f"{method}_{metric}" for metric in METRICS]
        parts.append(selected)
    return pd.concat(parts, axis=1).reset_index()


def analyse(
    merged: pd.DataFrame,
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for comparator in COMPARATORS:
        for metric in METRICS:
            comparator_values = merged[
                f"{comparator}_{metric}"
            ].to_numpy(dtype=float)
            prap_values = merged[f"prap_{metric}"].to_numpy(dtype=float)

            if metric in HIGHER_IS_BETTER:
                improvements = prap_values - comparator_values
                direction = "PRAP - comparator (higher is better)"
            else:
                improvements = comparator_values - prap_values
                direction = "comparator - PRAP (lower is better)"

            statistic, p_value = wilcoxon_paired(improvements)
            ci_low, ci_high = paired_bootstrap_ci(
                improvements,
                repeats=bootstrap_repeats,
                rng=rng,
            )

            row: dict[str, float | int | str] = {
                "Comparison": f"PRAP vs {comparator}",
                "Metric": metric,
                "N": int(improvements.size),
                "Nonzero_Pairs": int(np.count_nonzero(improvements)),
                "Comparator_Mean": float(comparator_values.mean()),
                "PRAP_Mean": float(prap_values.mean()),
                "Mean_Improvement": float(improvements.mean()),
                "Median_Improvement": float(np.median(improvements)),
                "CI95_Low": ci_low,
                "CI95_High": ci_high,
                "Wilcoxon_W": statistic,
                "P_Raw": p_value,
                "Rank_Biserial": paired_rank_biserial(improvements),
                "Improvement_Definition": direction,
            }

            if metric == "FP":
                row["Comparator_Total"] = int(comparator_values.sum())
                row["PRAP_Total"] = int(prap_values.sum())
                row["Total_Reduction"] = int(
                    comparator_values.sum() - prap_values.sum()
                )
            else:
                row["Comparator_Total"] = ""
                row["PRAP_Total"] = ""
                row["Total_Reduction"] = ""

            rows.append(row)

    results = pd.DataFrame(rows)
    results["P_Holm_All24"] = holm_adjust(
        results["P_Raw"].to_numpy(dtype=float)
    )

    results["P_Holm_Within_Metric"] = np.nan
    for metric, indices in results.groupby("Metric").groups.items():
        metric_p = results.loc[indices, "P_Raw"].to_numpy(dtype=float)
        results.loc[indices, "P_Holm_Within_Metric"] = holm_adjust(
            metric_p
        )

    return results


def write_text_summary(results: pd.DataFrame, output_path: Path) -> None:
    key = results.loc[results["Comparison"] == "PRAP vs none"].copy()
    key = key.set_index("Metric")

    lines = [
        "PRAP paired statistical summary (test set)",
        "==========================================",
        "",
        "Positive mean improvement always favors PRAP.",
        "Holm correction across all 24 tests is reported.",
        "",
    ]

    for metric in METRICS:
        row = key.loc[metric]
        lines.append(
            f"{metric}: comparator mean={row['Comparator_Mean']:.6f}, "
            f"PRAP mean={row['PRAP_Mean']:.6f}, "
            f"mean improvement={row['Mean_Improvement']:.6f}, "
            f"95% CI=[{row['CI95_Low']:.6f}, "
            f"{row['CI95_High']:.6f}], "
            f"Holm p={row['P_Holm_All24']:.6g}, "
            f"rank-biserial={row['Rank_Biserial']:.6f}"
        )

    fp_row = key.loc["FP"]
    lines.extend(
        [
            "",
            "FP totals:",
            f"None={int(fp_row['Comparator_Total'])}, "
            f"PRAP={int(fp_row['PRAP_Total'])}, "
            f"reduction={int(fp_row['Total_Reduction'])}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {name: read_csv(path) for name, path in DEFAULT_FILES.items()}
    none = validate_none_rows(frames)
    area = method_rows(frames["area"], "area")
    entropy = method_rows(frames["entropy"], "entropy")
    seeded = method_rows(frames["seeded"], "seeded")
    prap = method_rows(frames["prap"], "prap")

    merged = build_merged_table(
        none=none,
        area=area,
        entropy=entropy,
        seeded=seeded,
        prap=prap,
    )
    results = analyse(
        merged=merged,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )

    merged_path = output_dir / "merged_test_per_image.csv"
    statistics_path = output_dir / "paired_statistics.csv"
    summary_path = output_dir / "statistical_summary.txt"

    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    results.to_csv(statistics_path, index=False, encoding="utf-8-sig")
    write_text_summary(results, summary_path)

    display_columns = [
        "Comparison",
        "Metric",
        "Mean_Improvement",
        "CI95_Low",
        "CI95_High",
        "P_Holm_All24",
        "Rank_Biserial",
    ]
    print("\nPaired statistical results")
    print("Positive improvement favors PRAP.\n")
    print(
        results.loc[:, display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )
    print(f"\nSaved: {merged_path}")
    print(f"Saved: {statistics_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()