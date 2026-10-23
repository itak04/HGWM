"""Quantify repetitive exploration from HGWM/WMNav episode logs.

The script consumes the existing ``df_results.pkl`` files. Newer logs also
contain ``selected_frontier_location`` and therefore support frontier-revisit
counts; older logs still support pose revisit and redundant-path metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_id: str
    steps: int
    path_length_m: float
    revisit_ratio: float
    redundant_path_ratio: float
    loop_closure_count: int
    unique_coverage_m2: float
    frontier_revisit_count: int | None


def _point(value: Any) -> tuple[float, float] | None:
    """Return the horizontal (x, z) coordinates from common log encodings."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().strip("[]()")
        try:
            values = [float(part) for part in cleaned.replace(",", " ").split()]
        except ValueError:
            return None
    else:
        try:
            values = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    if len(values) >= 3:
        return values[0], values[2]
    if len(values) == 2:
        return values[0], values[1]
    return None


def _cell(point: tuple[float, float], cell_size_m: float) -> tuple[int, int]:
    return (math.floor(point[0] / cell_size_m), math.floor(point[1] / cell_size_m))


def _revisit_flags(
    points: Sequence[tuple[float, float]],
    cell_size_m: float,
    min_gap_steps: int,
) -> list[bool]:
    last_seen: dict[tuple[int, int], int] = {}
    flags: list[bool] = []
    for index, point in enumerate(points):
        cell = _cell(point, cell_size_m)
        previous = last_seen.get(cell)
        flags.append(previous is not None and index - previous >= min_gap_steps)
        last_seen[cell] = index
    return flags


def compute_episode_metrics(
    rows: Iterable[Mapping[str, Any]],
    episode_id: str = "episode",
    cell_size_m: float = 0.5,
    min_gap_steps: int = 2,
) -> EpisodeMetrics:
    """Compute explicitly defined loop/revisit metrics for one episode.

    ``revisit_ratio`` is the fraction of logged poses whose 0.5 m cell was
    visited at least ``min_gap_steps`` earlier. ``redundant_path_ratio`` is the
    fraction of traveled distance ending in such revisited cells. A loop
    closure is the first revisited pose in a contiguous revisit run.
    """
    row_list = list(rows)
    positions = [_point(row.get("agent_location")) for row in row_list]
    positions = [point for point in positions if point is not None]
    revisit_flags = _revisit_flags(positions, cell_size_m, min_gap_steps)

    segment_lengths = [
        math.dist(positions[index - 1], positions[index])
        for index in range(1, len(positions))
    ]
    path_length = sum(segment_lengths)
    revisited_length = sum(
        length
        for index, length in enumerate(segment_lengths, start=1)
        if revisit_flags[index]
    )
    loop_closures = sum(
        flag and (index == 0 or not revisit_flags[index - 1])
        for index, flag in enumerate(revisit_flags)
    )

    frontier_points = [
        _point(row.get("selected_frontier_location")) for row in row_list
    ]
    frontier_points = [point for point in frontier_points if point is not None]
    if frontier_points:
        frontier_flags = _revisit_flags(frontier_points, cell_size_m, min_gap_steps)
        frontier_revisits: int | None = sum(
            flag and (index == 0 or not frontier_flags[index - 1])
            for index, flag in enumerate(frontier_flags)
        )
    else:
        frontier_revisits = None

    unique_cells = {_cell(point, cell_size_m) for point in positions}
    return EpisodeMetrics(
        episode_id=episode_id,
        steps=len(positions),
        path_length_m=path_length,
        revisit_ratio=(sum(revisit_flags) / len(revisit_flags)) if revisit_flags else 0.0,
        redundant_path_ratio=(revisited_length / path_length) if path_length > 0 else 0.0,
        loop_closure_count=int(loop_closures),
        unique_coverage_m2=len(unique_cells) * cell_size_m * cell_size_m,
        frontier_revisit_count=frontier_revisits,
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_difference(
    treatment: Sequence[float],
    baseline: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return treatment-minus-baseline mean, paired 95% CI and sign-flip p."""
    if len(treatment) != len(baseline) or not treatment:
        raise ValueError("paired samples must be non-empty and have equal length")
    differences = [left - right for left, right in zip(treatment, baseline)]
    rng = random.Random(seed)
    bootstrap_means = [
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(n_resamples)
    ]
    observed = abs(sum(differences) / len(differences))
    randomized = [
        abs(sum(diff * rng.choice((-1.0, 1.0)) for diff in differences) / len(differences))
        for _ in range(n_resamples)
    ]
    return {
        "n": len(differences),
        "mean_difference": sum(differences) / len(differences),
        "ci95_low": _percentile(bootstrap_means, 0.025),
        "ci95_high": _percentile(bootstrap_means, 0.975),
        "randomization_p": (1 + sum(value >= observed for value in randomized))
        / (n_resamples + 1),
    }


def load_run(root: Path, cell_size_m: float, min_gap_steps: int) -> dict[str, EpisodeMetrics]:
    import pandas as pd

    result: dict[str, EpisodeMetrics] = {}
    for path in sorted(root.rglob("df_results.pkl")):
        frame = pd.read_pickle(path)
        episode_id = path.parent.name
        result[episode_id] = compute_episode_metrics(
            frame.to_dict("records"),
            episode_id=episode_id,
            cell_size_m=cell_size_m,
            min_gap_steps=min_gap_steps,
        )
    if not result:
        raise FileNotFoundError(f"no df_results.pkl files found under {root}")
    return result


def _mean(items: Sequence[float]) -> float:
    return sum(items) / len(items) if items else float("nan")


def summarize(run: Mapping[str, EpisodeMetrics]) -> dict[str, float | int | None]:
    metrics = list(run.values())
    frontier = [item.frontier_revisit_count for item in metrics if item.frontier_revisit_count is not None]
    return {
        "episodes": len(metrics),
        "revisit_ratio": _mean([item.revisit_ratio for item in metrics]),
        "redundant_path_ratio": _mean([item.redundant_path_ratio for item in metrics]),
        "loop_closure_count": _mean([float(item.loop_closure_count) for item in metrics]),
        "unique_coverage_m2": _mean([item.unique_coverage_m2 for item in metrics]),
        "frontier_revisit_count": _mean([float(value) for value in frontier]) if frontier else None,
    }


def compare_runs(
    treatment: Mapping[str, EpisodeMetrics],
    baseline: Mapping[str, EpisodeMetrics],
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    common = sorted(set(treatment) & set(baseline))
    if not common:
        raise ValueError("runs have no paired episode identifiers in common")
    output: dict[str, dict[str, float | int]] = {}
    for field in ("revisit_ratio", "redundant_path_ratio", "loop_closure_count", "unique_coverage_m2"):
        output[field] = paired_bootstrap_difference(
            [float(getattr(treatment[key], field)) for key in common],
            [float(getattr(baseline[key], field)) for key in common],
            n_resamples=n_resamples,
            seed=seed,
        )
    frontier_common = [
        key for key in common
        if treatment[key].frontier_revisit_count is not None
        and baseline[key].frontier_revisit_count is not None
    ]
    if frontier_common:
        output["frontier_revisit_count"] = paired_bootstrap_difference(
            [float(treatment[key].frontier_revisit_count) for key in frontier_common],
            [float(baseline[key].frontier_revisit_count) for key in frontier_common],
            n_resamples=n_resamples,
            seed=seed,
        )
    return output


def _parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must use NAME=LOG_ROOT")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", type=_parse_method, required=True)
    parser.add_argument("--baseline", help="method name used for paired comparisons")
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("behavioral_metrics.json"))
    args = parser.parse_args()

    runs = {
        name: load_run(path, args.cell_size, args.min_gap)
        for name, path in args.method
    }
    report: dict[str, Any] = {
        "definitions": {
            "cell_size_m": args.cell_size,
            "min_gap_steps": args.min_gap,
            "difference_direction": "method minus baseline",
        },
        "summary": {name: summarize(run) for name, run in runs.items()},
        "episodes": {
            name: {key: asdict(value) for key, value in run.items()}
            for name, run in runs.items()
        },
    }
    if args.baseline:
        if args.baseline not in runs:
            raise ValueError(f"unknown baseline: {args.baseline}")
        report["paired_comparisons"] = {
            name: compare_runs(run, runs[args.baseline], args.resamples, args.seed)
            for name, run in runs.items()
            if name != args.baseline
        }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
