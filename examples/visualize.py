import argparse
import csv
import os
import subprocess

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style("white")

BASELINE_COLOR = "#0073ff"
VISIONRT_COLOR = "#FF8C00"


def _clip_outliers(data_ms: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if data_ms.size == 0:
        return data_ms
    mu = data_ms.mean()
    sd = data_ms.std()
    if sd == 0.0:
        return data_ms
    mask = np.abs(data_ms - mu) <= sigma * sd
    return data_ms[mask]


def _get_nvtx_durations_ms(rep_path: str, target_names: set[str]) -> np.ndarray:
    proc = subprocess.run(
        [
            "nsys",
            "stats",
            "--format",
            "csv",
            "--force-export=true",
            "-r",
            "nvtx_pushpop_trace",
            rep_path,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"Warning: nsys stats exited with code {proc.returncode} for {rep_path}")
        return np.array([], dtype=np.float64)
    lines = proc.stdout.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Start (ns),End (ns),Duration (ns)"):
            header_idx = i
            break

    if header_idx is None:
        return np.array([], dtype=np.float64)

    csv_lines = lines[header_idx:]
    reader = csv.reader(csv_lines)

    try:
        header = next(reader)
    except StopIteration:
        return np.array([], dtype=np.float64)

    try:
        dur_idx = header.index("Duration (ns)")
        name_idx = header.index("Name")
    except ValueError:
        return np.array([], dtype=np.float64)

    durations_ns = []
    for row in reader:
        if len(row) <= max(dur_idx, name_idx):
            continue
        name = row[name_idx]
        if name not in target_names:
            continue
        try:
            durations_ns.append(float(row[dur_idx]))
        except ValueError:
            continue

    if not durations_ns:
        return np.array([], dtype=np.float64)

    durations_ms = np.array(durations_ns, dtype=np.float64) / 1e6
    return durations_ms


def _run_nsys(iters: int | None = None) -> None:
    env = os.environ.copy()
    if iters is not None:
        env["ITERS"] = str(iters)

    cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda,osrt,nvtx",
        "--force-overwrite=true",
        "-o",
        "e2e",
        "python3",
        "./examples/profile_e2e.py",
    ]

    proc = subprocess.run(cmd, check=False, env=env)

    if proc.returncode != 0:
        print(
            f"Warning: nsys command exited with code {proc.returncode}: "
            + " ".join(cmd)
        )


def main(iters: int = 100) -> None:
    # _run_nsys(iters) # uncomment if the the nsys-rep is not already generated

    baseline_ms = _get_nvtx_durations_ms("e2e.nsys-rep", {":baseline_e2e"})
    visionrt_ms = _get_nvtx_durations_ms("e2e.nsys-rep", {":visionrt_e2e"})

    baseline_ms = _clip_outliers(baseline_ms, sigma=3.0)
    visionrt_ms = _clip_outliers(visionrt_ms, sigma=3.0)

    labels = (["Baseline"] * len(baseline_ms)) + (["VisionRT"] * len(visionrt_ms))
    latencies = np.concatenate([baseline_ms, visionrt_ms])

    fig, ax = plt.subplots(figsize=(12, 6))

    df = pd.DataFrame({"Latency": latencies, "System": labels})

    sns.histplot(
        data=df,
        x="Latency",
        hue="System",
        kde=False,
        bins=40,
        alpha=1.0,
        common_norm=False,
        palette={"Baseline": BASELINE_COLOR, "VisionRT": VISIONRT_COLOR},
        stat="percent",
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    baseline_mean = baseline_ms.mean()
    baseline_std = baseline_ms.std()

    visionrt_mean = visionrt_ms.mean()
    visionrt_std = visionrt_ms.std()

    ax.axvline(baseline_mean, color="black", linestyle="--", linewidth=2)
    ax.axvline(visionrt_mean, color="black", linestyle="--", linewidth=2)

    ax.text(
        baseline_mean + 1.5,
        ax.get_ylim()[1] * 0.30,
        f"µ = {baseline_mean:.1f}ms\nσ = {baseline_std:.1f}ms",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
    )
    ax.text(
        visionrt_mean + 1.5,
        ax.get_ylim()[1] * 0.85,
        f"µ = {visionrt_mean:.1f}ms\nσ = {visionrt_std:.1f}ms",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    ax.set_xlabel("Per-frame Latency (ms)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percent", fontsize=14, fontweight="bold")
    ax.set_title(
        "Latency Distribution",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )

    legend_elements = [
        Patch(facecolor=BASELINE_COLOR, edgecolor="black", linewidth=0.8, label="Baseline"),
        Patch(facecolor=VISIONRT_COLOR, edgecolor="black", linewidth=0.8, label="VisionRT"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=13,
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="lightgray",
        handlelength=1.5,
        handleheight=1.5,
    )

    ax.grid(axis="y", alpha=0.3, linestyle="-", color="lightgray")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    plt.tight_layout()

    fig.savefig("images/latency_histogram.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of iterations to use for latency visualization.",
    )
    args = parser.parse_args()
    main(iters=args.iters)
