from __future__ import annotations

import argparse
import csv
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_postprocess import (
    DEFAULT_CHECKPOINT,
    REPO_ROOT,
    build_loader,
    choose_names,
    load_checkpoint,
)
from postprocess import PostprocessParams, apply_postprocess
from train_common import IMAGE_SIZE, UnetBaseline, set_seed


SCRIPT_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runtime_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark frozen Baseline inference and PRAP post-processing. "
            "Disk I/O, metric calculation, and CSV writing are excluded."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of test images used for warm-up.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of full-test-set timing repeats.",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.50,
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
        "--min-area",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def require_valid_args(args: argparse.Namespace, image_count: int) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if image_count <= 0:
        raise ValueError("No test images were loaded.")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def windows_cim_value(class_name: str, property_name: str) -> str:
    if platform.system() != "Windows":
        return ""
    command = (
        f"(Get-CimInstance {class_name} | "
        f"Select-Object -First 1 -ExpandProperty {property_name})"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def system_information(device: torch.device) -> dict[str, str]:
    cpu_name = windows_cim_value("Win32_Processor", "Name")
    if not cpu_name:
        cpu_name = platform.processor() or platform.machine()

    memory_bytes = windows_cim_value(
        "Win32_ComputerSystem",
        "TotalPhysicalMemory",
    )
    if memory_bytes.isdigit():
        memory_gb = f"{int(memory_bytes) / (1024 ** 3):.1f} GB"
    else:
        memory_gb = "unknown"

    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU only"
    )

    return {
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA_Runtime": str(torch.version.cuda),
        "Device": str(device),
        "GPU": gpu_name,
        "CPU": cpu_name,
        "RAM": memory_gb,
        "Image_Size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "Batch_Size": "1",
    }


def preload_test_images() -> list[torch.Tensor]:
    """
    Load the fixed test split into CPU memory before timing.

    Data decoding, resizing, and DataLoader overhead are intentionally excluded
    from the reported model and PRAP times.
    """
    names = choose_names("test")
    _, loader = build_loader(names, batch_size=1)

    images: list[torch.Tensor] = []
    for batch_images, _ in loader:
        if batch_images.shape[0] != 1:
            raise RuntimeError("Runtime benchmark requires batch_size=1.")
        images.append(batch_images.contiguous().float())

    if len(images) != len(names):
        raise RuntimeError(
            f"Loaded {len(images)} images, expected {len(names)}."
        )
    return images


def warm_up_model(
    model: torch.nn.Module,
    cpu_images: list[torch.Tensor],
    device: torch.device,
    warmup_count: int,
) -> None:
    if warmup_count == 0:
        return

    model.eval()
    count = min(warmup_count, len(cpu_images))
    with torch.inference_mode():
        for index in range(count):
            image = cpu_images[index].to(device)
            _ = torch.sigmoid(model(image))
    synchronize(device)


def benchmark_model_inference(
    model: torch.nn.Module,
    cpu_images: list[torch.Tensor],
    device: torch.device,
    repeats: int,
) -> list[float]:
    """
    Time forward pass + sigmoid only.

    The input tensor is transferred to the device before timing. Disk I/O,
    preprocessing, host-to-device transfer, post-processing, and metrics are
    excluded.
    """
    model.eval()
    repeat_means: list[float] = []

    with torch.inference_mode():
        for _ in range(repeats):
            image_times_ms: list[float] = []

            for cpu_image in cpu_images:
                image = cpu_image.to(device)
                synchronize(device)

                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    _ = torch.sigmoid(model(image))
                    end_event.record()

                    end_event.synchronize()
                    elapsed_ms = float(start_event.elapsed_time(end_event))
                else:
                    start = time.perf_counter()
                    _ = torch.sigmoid(model(image))
                    elapsed_ms = (time.perf_counter() - start) * 1000.0

                image_times_ms.append(elapsed_ms)

            repeat_means.append(float(np.mean(image_times_ms)))

    return repeat_means


def precompute_probability_maps(
    model: torch.nn.Module,
    cpu_images: list[torch.Tensor],
    device: torch.device,
) -> list[np.ndarray]:
    """
    Generate probability maps once, outside the PRAP timing section.
    """
    maps: list[np.ndarray] = []
    model.eval()

    with torch.inference_mode():
        for cpu_image in cpu_images:
            image = cpu_image.to(device)
            probability = torch.sigmoid(model(image))
            probability_np = (
                probability[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
            maps.append(probability_np)

    synchronize(device)
    return maps


def warm_up_prap(
    probability_maps: list[np.ndarray],
    params: PostprocessParams,
    warmup_count: int,
) -> None:
    count = min(warmup_count, len(probability_maps))
    for index in range(count):
        _ = apply_postprocess(
            "prap",
            probability_maps[index],
            params,
        )


def benchmark_prap(
    probability_maps: list[np.ndarray],
    params: PostprocessParams,
    repeats: int,
) -> list[float]:
    """
    Time the formal PRAP function on CPU probability maps.

    Model inference, tensor transfer, disk I/O, and metric calculation are
    excluded.
    """
    repeat_means: list[float] = []

    for _ in range(repeats):
        image_times_ms: list[float] = []

        for probability_map in probability_maps:
            start = time.perf_counter()
            _ = apply_postprocess(
                "prap",
                probability_map,
                params,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            image_times_ms.append(elapsed_ms)

        repeat_means.append(float(np.mean(image_times_ms)))

    return repeat_means


def benchmark_compute_pipeline(
    model: torch.nn.Module,
    cpu_images: list[torch.Tensor],
    params: PostprocessParams,
    device: torch.device,
    repeats: int,
) -> list[float]:
    """
    Time model forward + sigmoid + device-to-host copy + PRAP.

    The input is moved to the device before timing. Data loading, image
    preprocessing, host-to-device transfer, metrics, and file writing are
    excluded.
    """
    model.eval()
    repeat_means: list[float] = []

    with torch.inference_mode():
        for _ in range(repeats):
            image_times_ms: list[float] = []

            for cpu_image in cpu_images:
                image = cpu_image.to(device)
                synchronize(device)

                start = time.perf_counter()

                probability = torch.sigmoid(model(image))
                probability_np = (
                    probability[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                _ = apply_postprocess(
                    "prap",
                    probability_np,
                    params,
                )

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                image_times_ms.append(elapsed_ms)

            repeat_means.append(float(np.mean(image_times_ms)))

    return repeat_means


def summarize(stage: str, values: list[float]) -> dict[str, float | str]:
    mean_value = float(statistics.mean(values))
    std_value = (
        float(statistics.stdev(values))
        if len(values) > 1
        else 0.0
    )
    return {
        "Stage": stage,
        "Mean_ms_per_image": mean_value,
        "Std_ms_per_image": std_value,
        "Min_repeat_mean_ms": float(min(values)),
        "Max_repeat_mean_ms": float(max(values)),
        "FPS_from_mean": 1000.0 / mean_value if mean_value > 0 else float("inf"),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    info: dict[str, str],
    rows: list[dict],
    overhead_percent: float,
    params: PostprocessParams,
    warmup: int,
    repeats: int,
    image_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"benchmark_runtime.py version: {SCRIPT_VERSION}",
        "",
        "Experimental conditions",
        f"- Test images: {image_count}",
        f"- Warm-up images: {warmup}",
        f"- Repeats: {repeats}",
        "- Batch size: 1",
        "- Excluded: disk I/O, image preprocessing, metric calculation, CSV writing",
        "- Model-only timing excludes host-to-device transfer",
        "- Pipeline timing includes forward, sigmoid, device-to-host copy, and PRAP",
        f"- PRAP parameters: {params}",
        "",
        "Hardware and software",
    ]
    lines.extend(f"- {key}: {value}" for key, value in info.items())

    lines.extend(["", "Results"])
    for row in rows:
        lines.append(
            f"- {row['Stage']}: "
            f"{row['Mean_ms_per_image']:.4f} ± "
            f"{row['Std_ms_per_image']:.4f} ms/image; "
            f"{row['FPS_from_mean']:.2f} images/s"
        )

    lines.extend(
        [
            "",
            (
                "PRAP relative overhead versus model-only inference: "
                f"{overhead_percent:.2f}%"
            ),
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"benchmark_runtime.py version: {SCRIPT_VERSION}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    cpu_images = preload_test_images()
    require_valid_args(args, len(cpu_images))
    print(f"Preloaded test images: {len(cpu_images)}")

    model = UnetBaseline(
        encoder_weights=None,
    ).to(device).float()
    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )
    model.eval()

    params = PostprocessParams(
        candidate_threshold=args.candidate_threshold,
        min_area=args.min_area,
        entropy_threshold=0.45,
        seed_threshold=args.seed_threshold,
        kernel_size=args.kernel_size,
        dilation_iterations=args.dilation_iterations,
    )

    print(f"PRAP parameters: {params}")
    print("Warming up model...")
    warm_up_model(
        model,
        cpu_images,
        device,
        args.warmup,
    )

    print("Benchmarking model inference...")
    model_times = benchmark_model_inference(
        model,
        cpu_images,
        device,
        args.repeats,
    )

    print("Precomputing probability maps outside PRAP timing...")
    probability_maps = precompute_probability_maps(
        model,
        cpu_images,
        device,
    )

    print("Warming up PRAP...")
    warm_up_prap(
        probability_maps,
        params,
        args.warmup,
    )

    print("Benchmarking PRAP post-processing...")
    prap_times = benchmark_prap(
        probability_maps,
        params,
        args.repeats,
    )

    print("Benchmarking compute-only pipeline...")
    pipeline_times = benchmark_compute_pipeline(
        model,
        cpu_images,
        params,
        device,
        args.repeats,
    )

    rows = [
        summarize("Baseline_forward_plus_sigmoid", model_times),
        summarize("PRAP_postprocessing_only", prap_times),
        summarize("Compute_pipeline", pipeline_times),
    ]

    model_mean = float(rows[0]["Mean_ms_per_image"])
    prap_mean = float(rows[1]["Mean_ms_per_image"])
    overhead_percent = (
        100.0 * prap_mean / model_mean
        if model_mean > 0
        else float("nan")
    )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    csv_path = output_dir / "runtime_benchmark.csv"
    summary_path = output_dir / "runtime_summary.txt"

    write_csv(rows, csv_path)
    write_summary(
        summary_path,
        system_information(device),
        rows,
        overhead_percent,
        params,
        args.warmup,
        args.repeats,
        len(cpu_images),
    )

    print("\nRuntime benchmark")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['Stage']:<34}"
            f"{row['Mean_ms_per_image']:>12.4f} ± "
            f"{row['Std_ms_per_image']:<10.4f} ms/image"
        )
    print("-" * 88)
    print(f"PRAP relative overhead: {overhead_percent:.2f}%")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()