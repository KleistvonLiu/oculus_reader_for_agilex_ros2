#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python oculus_reader/plot_t_matrix_log.py /home/kleist/Documents/Code/oculus_reader/oculus_reader/t_matrix_log_20260113_092638.csv

"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path: Path):
    rows = []
    with path.open("r", encoding="ascii") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def to_float_series(rows, key):
    return [float(r[key]) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Plot t_matrix_log CSV data.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="oculus_reader/t_matrix_log.csv",
        help="Path to t_matrix_log CSV file.",
    )
    args = parser.parse_args()

    path = Path(args.csv_path)
    rows = read_log(path)

    controllers = sorted({r.get("controller", "unknown") for r in rows})
    pose_types = sorted({r.get("type", "unknown") for r in rows})
    for controller in controllers:
        controller_rows = [r for r in rows if r.get("controller", "unknown") == controller]
        if not controller_rows:
            continue
        for pose_type in pose_types:
            type_rows = [
                r for r in controller_rows if r.get("type", "unknown") == pose_type
            ]
            if not type_rows:
                continue
            t = to_float_series(type_rows, "timestamp")
            t0 = t[0]
            t = [ti - t0 for ti in t]

            series = {
                "x": to_float_series(type_rows, "x"),
                "y": to_float_series(type_rows, "y"),
                "z": to_float_series(type_rows, "z"),
                "roll": to_float_series(type_rows, "roll"),
                "pitch": to_float_series(type_rows, "pitch"),
                "yaw": to_float_series(type_rows, "yaw"),
            }

            fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
            for i, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
                axes[i].plot(t, series[key], label=key)
                axes[i].set_title(key)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlabel("time (s)")
                unit = "m" if key in ("x", "y", "z") else "rad"
                axes[i].set_ylabel(unit)

            fig.suptitle(f"Controller: {controller} | Pose: {pose_type}", y=0.98)
            fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
