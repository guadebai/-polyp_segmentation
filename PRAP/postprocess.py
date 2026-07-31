from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostprocessParams:
    """Parameters shared by the post-processing methods."""

    candidate_threshold: float = 0.50
    min_area: int = 80
    entropy_threshold: float = 0.45
    seed_threshold: float = 0.95
    kernel_size: int = 3
    dilation_iterations: int = 5


def _validate_probability_map(probability: np.ndarray) -> np.ndarray:
    """Return a finite 2-D float32 probability map in [0, 1]."""
    array = np.asarray(probability, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D probability map, got shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Probability map contains NaN or infinite values.")

    return np.clip(array, 0.0, 1.0)


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _validate_min_area(min_area: int) -> None:
    if min_area < 0:
        raise ValueError(f"min_area must be non-negative, got {min_area}.")


def _validate_kernel(kernel_size: int, iterations: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd integer, "
            f"got {kernel_size}."
        )
    if iterations < 0:
        raise ValueError(
            "dilation_iterations must be non-negative, "
            f"got {iterations}."
        )


def _remove_small_components(
    binary_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """Keep connected foreground components with area >= min_area."""
    _validate_min_area(min_area)

    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)

    if min_area <= 1 or mask.max() == 0:
        return mask

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    output = np.zeros_like(mask, dtype=np.uint8)

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            output[labels == component_id] = 1

    return output


def no_postprocess(
    probability: np.ndarray,
    candidate_threshold: float = 0.50,
) -> np.ndarray:
    """Binarize the probability map without further filtering."""
    _validate_threshold("candidate_threshold", candidate_threshold)
    prob = _validate_probability_map(probability)
    return (prob >= candidate_threshold).astype(np.uint8)


def area_filter(
    probability: np.ndarray,
    candidate_threshold: float = 0.50,
    min_area: int = 80,
) -> np.ndarray:
    """Remove candidate connected components smaller than min_area."""
    candidate = no_postprocess(
        probability,
        candidate_threshold=candidate_threshold,
    )
    return _remove_small_components(candidate, min_area=min_area)


def binary_entropy(probability: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute pixel-wise binary Shannon entropy using the natural logarithm.

    The maximum entropy is ln(2), approximately 0.6931.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    prob = _validate_probability_map(probability)
    prob = np.clip(prob, eps, 1.0 - eps)

    return -(
        prob * np.log(prob)
        + (1.0 - prob) * np.log(1.0 - prob)
    )


def entropy_filter(
    probability: np.ndarray,
    candidate_threshold: float = 0.50,
    entropy_threshold: float = 0.45,
    min_area: int = 80,
) -> np.ndarray:
    """
    Keep candidate components whose mean binary entropy is low enough.

    The same minimum-area constraint is applied so that the comparison with
    area filtering and PRAP uses a common basic geometric constraint.
    """
    _validate_threshold("candidate_threshold", candidate_threshold)
    _validate_min_area(min_area)

    if entropy_threshold < 0.0:
        raise ValueError(
            "entropy_threshold must be non-negative, "
            f"got {entropy_threshold}."
        )

    prob = _validate_probability_map(probability)
    candidate = (prob >= candidate_threshold).astype(np.uint8)
    entropy_map = binary_entropy(prob)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    output = np.zeros_like(candidate, dtype=np.uint8)

    for component_id in range(1, component_count):
        component = labels == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        mean_entropy = float(entropy_map[component].mean())
        if mean_entropy <= entropy_threshold:
            output[component] = 1

    return output



def seeded_component_filter(
    probability: np.ndarray,
    candidate_threshold: float = 0.50,
    seed_threshold: float = 0.95,
    min_area: int = 0,
) -> np.ndarray:
    """
    Keep each complete candidate component only when it contains a seed.

    This is a strong double-threshold / hysteresis-style baseline:

        candidate = probability >= candidate_threshold
        seed = probability >= seed_threshold

    A complete candidate connected component is retained if at least one
    high-confidence seed pixel lies inside it. Unlike PRAP, this method does
    not crop the component around the seed-supported local region.

    Ground truth is never accepted or used by this function.
    """
    _validate_threshold("candidate_threshold", candidate_threshold)
    _validate_threshold("seed_threshold", seed_threshold)
    _validate_min_area(min_area)

    if seed_threshold < candidate_threshold:
        raise ValueError(
            "seed_threshold should be greater than or equal to "
            "candidate_threshold."
        )

    prob = _validate_probability_map(probability)
    candidate = (prob >= candidate_threshold).astype(np.uint8)
    seed = (prob >= seed_threshold).astype(np.uint8)

    if candidate.max() == 0 or seed.max() == 0:
        return np.zeros_like(candidate, dtype=np.uint8)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    output = np.zeros_like(candidate, dtype=np.uint8)

    for component_id in range(1, component_count):
        component = labels == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        if np.any(seed[component] > 0):
            output[component] = 1

    return output

def prap_filter(
    probability: np.ndarray,
    candidate_threshold: float = 0.50,
    seed_threshold: float = 0.95,
    kernel_size: int = 3,
    dilation_iterations: int = 5,
    min_area: int = 80,
) -> np.ndarray:
    """
    Apply the GT-free PRAP implementation used in the formal experiment.

    Steps
    -----
    1. Extract the candidate mask at candidate_threshold.
    2. Extract high-confidence seed pixels at seed_threshold.
    3. Dilate the seed map to obtain a retrieval mask.
    4. Keep only candidate pixels covered by the retrieval mask.
    5. Remove connected components smaller than min_area.

    Important
    ---------
    This implementation performs local anchor-supported cropping:

        anchored = candidate AND dilated_seed

    It does not retain an entire candidate component merely because one seed
    intersects it. This distinction must be described consistently in the
    manuscript.

    Ground truth is never accepted or used by this function.
    """
    _validate_threshold("candidate_threshold", candidate_threshold)
    _validate_threshold("seed_threshold", seed_threshold)
    _validate_kernel(kernel_size, dilation_iterations)
    _validate_min_area(min_area)

    if seed_threshold < candidate_threshold:
        raise ValueError(
            "seed_threshold should be greater than or equal to "
            "candidate_threshold."
        )

    prob = _validate_probability_map(probability)

    candidate = (prob >= candidate_threshold).astype(np.uint8)
    seed = (prob >= seed_threshold).astype(np.uint8)

    if seed.max() == 0:
        return np.zeros_like(candidate, dtype=np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    if dilation_iterations == 0:
        retrieval = seed
    else:
        retrieval = cv2.dilate(
            seed,
            kernel,
            iterations=dilation_iterations,
        )

    anchored = np.logical_and(
        candidate > 0,
        retrieval > 0,
    ).astype(np.uint8)

    return _remove_small_components(
        anchored,
        min_area=min_area,
    )


def apply_postprocess(
    method: str,
    probability: np.ndarray,
    params: PostprocessParams,
) -> np.ndarray:
    """Dispatch one of the four formal comparison methods."""
    method_key = method.strip().lower()

    if method_key == "none":
        return no_postprocess(
            probability,
            candidate_threshold=params.candidate_threshold,
        )

    if method_key == "area":
        return area_filter(
            probability,
            candidate_threshold=params.candidate_threshold,
            min_area=params.min_area,
        )

    if method_key == "entropy":
        return entropy_filter(
            probability,
            candidate_threshold=params.candidate_threshold,
            entropy_threshold=params.entropy_threshold,
            min_area=params.min_area,
        )

    if method_key == "seeded":
        return seeded_component_filter(
            probability,
            candidate_threshold=params.candidate_threshold,
            seed_threshold=params.seed_threshold,
            min_area=params.min_area,
        )

    if method_key == "prap":
        return prap_filter(
            probability,
            candidate_threshold=params.candidate_threshold,
            seed_threshold=params.seed_threshold,
            kernel_size=params.kernel_size,
            dilation_iterations=params.dilation_iterations,
            min_area=params.min_area,
        )

    raise ValueError(
        f"Unknown post-processing method: {method}. "
        "Available methods: none, area, entropy, seeded, prap."
    )