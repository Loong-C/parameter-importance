#!/usr/bin/env python3
"""Build the Stage 2 estimator analysis tables and figures from immutable JSON outputs.

The script consumes the direct real-Pythia run.  It never edits the source
artifacts.  Every plotted accuracy value is reloaded from the batch summary
identified by the hash stored in ``s2.7-main-sweep.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


METHOD_ORDER = ["raw", "double", "u_m2", "u_m4", "u_m8", "u_m16", "u_m32"]
METHOD_LABEL = {
    "raw": "Raw",
    "double": "Double",
    "u_m2": "U-2",
    "u_m4": "U-4",
    "u_m8": "U-8",
    "u_m16": "U-16",
    "u_m32": "U-32",
}
METHOD_COLORS = {
    "raw": "#DE8F05",
    "double": "#9B5AA0",
    "u_m2": "#8FB8D0",
    "u_m4": "#5B9BC4",
    "u_m8": "#2F7FB5",
    "u_m16": "#1769A0",
    "u_m32": "#064C7D",
}
METHOD_MARKERS = {
    "raw": "o",
    "double": "s",
    "u_m2": "D",
    "u_m4": "^",
    "u_m8": "v",
    "u_m16": "P",
    "u_m32": "X",
}
METHOD_LINESTYLES = {
    "raw": "-",
    "double": "--",
    "u_m2": ":",
    "u_m4": "-.",
    "u_m8": "-",
    "u_m16": "--",
    "u_m32": ":",
}
BATCH_COLORS = {32: "#0173B2", 64: "#56B4E9", 128: "#DE8F05", 256: "#9B5AA0"}
STAGE_ORDER = ["initialization", "early", "mid_late"]
STAGE_LABEL = {"initialization": "初始化", "early": "早期", "mid_late": "中后期"}
MODEL_ORDER = ["pythia-14m", "pythia-31m-deduped"]
MODEL_LABEL = {"pythia-14m": "Pythia-14M", "pythia-31m-deduped": "Pythia-31M"}
CELL_ORDER = [f"{model}:{stage}" for model in MODEL_ORDER for stage in STAGE_ORDER]
CELL_LABEL = {
    cell: f"{MODEL_LABEL[cell.split(':', 1)[0]]} · {STAGE_LABEL[cell.split(':', 1)[1]]}"
    for cell in CELL_ORDER
}

INK = "#24292F"
MUTED = "#667085"
GRID = "#D9DEE7"
BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
PURPLE = "#9B5AA0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def split_cell(cell_id: str) -> tuple[str, str]:
    model, stage = cell_id.split(":", 1)
    return model, stage


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#AAB2BF",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )


def figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=0.985, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.055, 0.945, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)


def figure_footer(fig: plt.Figure) -> None:
    fig.text(
        0.055,
        0.018,
        "来源：Stage 2 direct real-Pythia run · pythia-grid-20260826T145530Z",
        fontsize=7.8,
        color=MUTED,
        ha="left",
    )


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(figures / f"{stem}.{suffix}", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def load_source_tables(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    delivery = input_root / "delivery-final-v2"
    main_path = delivery / "s2.7-main-sweep.json"
    profiler_path = delivery / "s2.9-profiler.json"
    complete_path = delivery / "complete.json"
    main = read_json(main_path)
    profiler = read_json(profiler_path)
    complete = read_json(complete_path)

    candidates = pd.DataFrame(main["candidate_rows"])
    if len(candidates) != 168:
        raise ValueError(f"Expected 168 candidate rows, found {len(candidates)}")
    if set(candidates["method"]) != set(METHOD_ORDER):
        raise ValueError("Estimator method coverage mismatch")
    if set(candidates["batch_size"].astype(int)) != {32, 64, 128, 256}:
        raise ValueError("Batch-size coverage mismatch")
    if set(candidates["cell_id"]) != set(CELL_ORDER):
        raise ValueError("Cell coverage mismatch")

    batch_by_hash: dict[str, dict[str, Any]] = {}
    summary_paths: dict[str, str] = {}
    for path in sorted(input_root.glob("*/batch-summaries.jsonl")):
        for payload in read_jsonl(path):
            artifact_hash = str(payload["artifact_hash"])
            if artifact_hash in batch_by_hash:
                raise ValueError(f"Duplicate batch artifact hash: {artifact_hash}")
            batch_by_hash[artifact_hash] = payload
            summary_paths[artifact_hash] = path.relative_to(input_root).as_posix()

    enriched: list[dict[str, Any]] = []
    metric_fields = ("mse", "mae", "pearson")
    for row in candidates.to_dict(orient="records"):
        source_hash = str(row["source_batch_artifact_hash"])
        if source_hash not in batch_by_hash:
            raise ValueError(f"Missing batch summary {source_hash}")
        batch = batch_by_hash[source_hash]
        if batch["cell_id"] != row["cell_id"] or int(batch["batch_size"]) != int(row["batch_size"]):
            raise ValueError("Candidate row and batch summary identity mismatch")
        method_payload = batch["methods"][row["method"]]
        if method_payload["artifact_hash"] != row["source_method_artifact_hash"]:
            raise ValueError("Candidate row and method artifact hash mismatch")
        reference = method_payload["reference_metrics_of_mean"]
        for field in metric_fields:
            if not math.isclose(float(reference[field]), float(row[field]), rel_tol=1e-12, abs_tol=1e-30):
                raise ValueError(f"Metric mismatch for {row['cell_id']} {row['method']} B={row['batch_size']} {field}")

        parameter_summaries = method_payload["parameter_summaries"]
        total_coordinates = sum(int(item["coordinate_count"]) for item in parameter_summaries)
        variance = sum(
            float(item["between_repetition_variance_mean"]) * int(item["coordinate_count"])
            for item in parameter_summaries
        ) / total_coordinates
        model, stage = split_cell(str(row["cell_id"]))
        enriched.append(
            {
                **row,
                "batch_size": int(row["batch_size"]),
                "successful_repetitions": int(batch["successful_repetitions"]),
                "model": model,
                "stage": stage,
                "model_label": MODEL_LABEL[model],
                "stage_label": STAGE_LABEL[stage],
                "cell_label": CELL_LABEL[row["cell_id"]],
                "method_label": METHOD_LABEL[row["method"]],
                "signed_bias_mean": float(reference["signed_bias_mean"]),
                "between_repetition_variance_mean": float(variance),
                "mean_negative_fraction": float(method_payload["mean_vector"]["negative_fraction"]),
                "mean_absolute_importance": float(method_payload["mean_vector"]["absolute_mean"]),
                "coordinate_count": int(method_payload["mean_vector"]["coordinate_count"]),
                "resolved_batch_summary_ref": summary_paths[source_hash],
            }
        )
    candidate_df = pd.DataFrame(enriched)

    profiler_df = pd.DataFrame(profiler["rows"])
    if len(profiler_df) != 24:
        raise ValueError(f"Expected 24 profiler rows, found {len(profiler_df)}")
    profiler_df["batch_size"] = profiler_df["batch_size"].astype(int)
    profiler_df[["model", "stage"]] = profiler_df["cell_id"].apply(
        lambda value: pd.Series(split_cell(value))
    )
    profiler_df["model_label"] = profiler_df["model"].map(MODEL_LABEL)
    profiler_df["stage_label"] = profiler_df["stage"].map(STAGE_LABEL)
    profiler_df["cell_label"] = profiler_df["cell_id"].map(CELL_LABEL)
    profiler_df["peak_memory_gib"] = profiler_df["peak_memory_bytes_max"] / (1024**3)
    profiler_df["formula_share"] = profiler_df["formula_seconds_mean"] / profiler_df["wall_seconds_mean"]

    metadata = {
        "run_id": main["run_id"],
        "scope": main["scope"],
        "formal_eligible": bool(main["formal_eligible"]),
        "main_sweep_artifact_hash": main["artifact_hash"],
        "profiler_artifact_hash": profiler["artifact_hash"],
        "delivery_complete_artifact_hash": complete["artifact_hash"],
        "delivery_hash": complete["delivery_hash"],
        "source_files": {
            main_path.relative_to(input_root).as_posix(): sha256_file(main_path),
            profiler_path.relative_to(input_root).as_posix(): sha256_file(profiler_path),
            complete_path.relative_to(input_root).as_posix(): sha256_file(complete_path),
        },
    }
    return candidate_df, profiler_df, metadata


def add_derived_metrics(candidate_df: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_df.copy()
    # Rank on rounded log-MSE so numerically identical constructions (Double
    # and U-2 differ only at floating-point roundoff) receive the same rank.
    frame["mse_rank_key"] = np.round(np.log10(frame["mse"]), 12)
    frame["mse_rank_within_cell_batch"] = frame.groupby(["cell_id", "batch_size"])["mse_rank_key"].rank(
        method="average", ascending=True
    )
    raw = (
        frame.loc[frame["method"] == "raw", ["cell_id", "batch_size", "mse"]]
        .rename(columns={"mse": "raw_mse"})
        .set_index(["cell_id", "batch_size"])
    )
    frame = frame.join(raw, on=["cell_id", "batch_size"])
    frame["mse_ratio_to_raw"] = frame["mse"] / frame["raw_mse"]
    frame["log10_mse_ratio_to_raw"] = np.log10(frame["mse_ratio_to_raw"])
    cell_min = frame.groupby("cell_id")["mse"].transform("min")
    frame["mse_ratio_to_cell_best"] = frame["mse"] / cell_min
    frame["log10_mse"] = np.log10(frame["mse"])
    return frame


def build_summary_tables(candidate_df: pd.DataFrame, profiler_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    method_summary = (
        candidate_df.groupby("method", sort=False)
        .agg(
            mean_mse_rank=("mse_rank_within_cell_batch", "mean"),
            median_mse_rank=("mse_rank_within_cell_batch", "median"),
            rank1_contexts=("mse_rank_within_cell_batch", lambda values: int(np.sum(np.isclose(values, 1.0)))),
            median_pearson=("pearson", "median"),
            median_abs_signed_bias=("signed_bias_mean", lambda values: float(np.median(np.abs(values)))),
            median_between_rep_variance=("between_repetition_variance_mean", "median"),
            geometric_mean_mse_ratio_to_raw=(
                "mse_ratio_to_raw",
                lambda values: float(np.exp(np.mean(np.log(values)))),
            ),
        )
        .reset_index()
    )
    method_summary["method_label"] = method_summary["method"].map(METHOD_LABEL)
    method_summary = method_summary.sort_values(["mean_mse_rank", "median_mse_rank", "method"]).reset_index(drop=True)

    recommendations = (
        candidate_df.sort_values(["cell_id", "mse", "mae", "pearson"], ascending=[True, True, True, False])
        .groupby("cell_id", as_index=False)
        .first()
    )
    recommendations["cell_order"] = recommendations["cell_id"].map({cell: i for i, cell in enumerate(CELL_ORDER)})
    recommendations = recommendations.sort_values("cell_order").drop(columns=["cell_order"])

    best_by_cell_method = (
        candidate_df.sort_values(["cell_id", "method", "mse", "batch_size"])
        .groupby(["cell_id", "method"], as_index=False)
        .first()
    )
    batch_frequency = (
        best_by_cell_method.groupby("batch_size")
        .size()
        .reindex([32, 64, 128, 256], fill_value=0)
        .rename("best_configuration_count")
        .reset_index()
    )
    batch_frequency["share"] = batch_frequency["best_configuration_count"] / len(best_by_cell_method)

    best_by_cell_batch = (
        candidate_df.sort_values(["cell_id", "batch_size", "mse", "mae", "pearson"], ascending=[True, True, True, True, False])
        .groupby(["cell_id", "batch_size"], as_index=False)
        .first()
    )
    best_by_cell_batch = best_by_cell_batch.merge(
        profiler_df[
            [
                "cell_id",
                "batch_size",
                "wall_seconds_mean",
                "gradient_seconds_mean",
                "formula_seconds_mean",
                "peak_memory_gib",
            ]
        ],
        on=["cell_id", "batch_size"],
        how="left",
        validate="one_to_one",
    )
    best_by_cell_batch["mse_ratio_to_cell_best"] = best_by_cell_batch["mse"] / best_by_cell_batch.groupby("cell_id")["mse"].transform("min")
    best_by_cell_batch["wall_ratio_to_cell_min"] = best_by_cell_batch["wall_seconds_mean"] / best_by_cell_batch.groupby("cell_id")["wall_seconds_mean"].transform("min")

    def grouped_method_summary(group_field: str) -> pd.DataFrame:
        return (
            candidate_df.groupby([group_field, "method"], sort=False)
            .agg(
                mean_mse_rank=("mse_rank_within_cell_batch", "mean"),
                geometric_mean_mse_ratio_to_raw=(
                    "mse_ratio_to_raw",
                    lambda values: float(np.exp(np.mean(np.log(values)))),
                ),
                median_pearson=("pearson", "median"),
            )
            .reset_index()
        )

    runtime_by_batch = (
        profiler_df.groupby("batch_size")
        .agg(
            median_wall_seconds=("wall_seconds_mean", "median"),
            minimum_wall_seconds=("wall_seconds_mean", "min"),
            maximum_wall_seconds=("wall_seconds_mean", "max"),
            median_peak_memory_gib=("peak_memory_gib", "median"),
            median_formula_share=("formula_share", "median"),
        )
        .reset_index()
    )
    b32_wall = float(runtime_by_batch.loc[runtime_by_batch["batch_size"] == 32, "median_wall_seconds"].iloc[0])
    runtime_by_batch["wall_ratio_to_b32"] = runtime_by_batch["median_wall_seconds"] / b32_wall

    return {
        "method_summary": method_summary,
        "recommendations": recommendations,
        "best_by_cell_method": best_by_cell_method,
        "batch_frequency": batch_frequency,
        "best_by_cell_batch": best_by_cell_batch,
        "stage_method_summary": grouped_method_summary("stage"),
        "model_method_summary": grouped_method_summary("model"),
        "runtime_by_batch": runtime_by_batch,
    }


def plot_coverage(candidate_df: pd.DataFrame, output_root: Path) -> None:
    counts = (
        candidate_df.groupby(["cell_id", "batch_size"])["successful_repetitions"]
        .nunique()
        .rename("unique_counts")
    )
    if int(counts.max()) != 1:
        raise ValueError("Method rows disagree on repetition count within a cell/batch")
    matrix = (
        candidate_df.groupby(["cell_id", "batch_size"])["successful_repetitions"]
        .first()
        .unstack("batch_size")
        .reindex(index=CELL_ORDER, columns=[32, 64, 128, 256])
    )
    cmap = LinearSegmentedColormap.from_list("coverage", ["#FCE8D3", "#E7F2F8", "#0173B2"])
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    figure_header(fig, "成功重复次数覆盖矩阵", "每格同时覆盖 7 种估计器；计划值为 64 次重复")
    image = ax.imshow(matrix.values, cmap=cmap, vmin=0, vmax=64, aspect="auto")
    ax.set_xticks(range(4), ["B=32", "B=64", "B=128", "B=256"])
    ax.set_yticks(range(6), [CELL_LABEL[cell] for cell in CELL_ORDER])
    ax.grid(False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix.iloc[row, col])
            color = "white" if value >= 50 else INK
            ax.text(col, row, f"n={value}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)
            if value < 64:
                ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor=ORANGE, linewidth=2.5))
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("成功重复次数")
    figure_footer(fig)
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.24, right=0.93)
    save_figure(fig, output_root, "fig01-coverage-matrix")


def plot_recommendation_map(recommendations: pd.DataFrame, output_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    figure_header(fig, "逐模型与训练阶段的最低 MSE 配置", "在每个真实 Pythia cell 的 28 个 method×B 候选中选择最低观测 MSE")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for model_index, model in enumerate(MODEL_ORDER):
        y = 1 - model_index
        for stage_index, stage in enumerate(STAGE_ORDER):
            cell = f"{model}:{stage}"
            row = recommendations.loc[recommendations["cell_id"] == cell].iloc[0]
            method = row["method"]
            color = METHOD_COLORS[method]
            rect = Rectangle((stage_index + 0.04, y + 0.08), 0.92, 0.84, facecolor=color, alpha=0.13, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(stage_index + 0.5, y + 0.69, METHOD_LABEL[method], ha="center", va="center", fontsize=15, fontweight="bold", color=color)
            ax.text(stage_index + 0.5, y + 0.45, f"B={int(row['batch_size'])}", ha="center", va="center", fontsize=11)
            ax.text(stage_index + 0.5, y + 0.25, f"MSE={row['mse']:.2e}", ha="center", va="center", fontsize=9.5, color=MUTED)
    for stage_index, stage in enumerate(STAGE_ORDER):
        ax.text(stage_index + 0.5, 2.03, STAGE_LABEL[stage], ha="center", va="bottom", fontsize=11, fontweight="bold")
    for model_index, model in enumerate(MODEL_ORDER):
        ax.text(-0.02, 1.5 - model_index, MODEL_LABEL[model], ha="right", va="center", fontsize=11, fontweight="bold")
    figure_footer(fig)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.16, right=0.98)
    save_figure(fig, output_root, "fig02-cell-winners")


def plot_method_rank(method_summary: pd.DataFrame, output_root: Path) -> None:
    ordered = method_summary.sort_values("mean_mse_rank", ascending=False)
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    figure_header(fig, "估计器在 24 个 cell×B 场景中的平均 MSE 排名", "横轴越小越好；右侧数字为并列第一的场景数")
    bars = ax.barh(
        ordered["method_label"],
        ordered["mean_mse_rank"],
        color=[METHOD_COLORS[m] for m in ordered["method"]],
        edgecolor="white",
        height=0.66,
    )
    ax.scatter(ordered["median_mse_rank"], np.arange(len(ordered)), marker="D", s=42, color=INK, zorder=3, label="中位排名")
    count_x = 5.35
    for bar, wins in zip(bars, ordered["rank1_contexts"]):
        ax.text(count_x, bar.get_y() + bar.get_height() / 2, f"第一名 {int(wins)} 次", va="center", fontsize=9.5)
    ax.axvline(1, color=INK, linestyle="--", linewidth=1)
    ax.set_xlabel("平均 MSE 排名（1 = 最优）")
    ax.set_xlim(0, 6.7)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    figure_footer(fig)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.16, right=0.96)
    save_figure(fig, output_root, "fig03-method-rank")


def plot_relative_mse(candidate_df: pd.DataFrame, output_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    figure_header(fig, "相对 Raw 的 MSE 分布", "每个点对应一个 cell×B 场景；0 表示与 Raw 相同，负值表示更低 MSE")
    values = [
        candidate_df.loc[candidate_df["method"] == method, "log10_mse_ratio_to_raw"].to_numpy()
        for method in METHOD_ORDER
    ]
    box = ax.boxplot(values, patch_artist=True, widths=0.58, showfliers=False, medianprops={"color": INK, "linewidth": 1.7})
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.28)
        patch.set_edgecolor(METHOD_COLORS[method])
        patch.set_linewidth(1.5)
    rng = np.random.default_rng(20260828)
    for index, (method, series) in enumerate(zip(METHOD_ORDER, values), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(series))
        ax.scatter(index + jitter, series, s=23, color=METHOD_COLORS[method], alpha=0.62, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color=INK, linewidth=1.2, linestyle="--")
    ax.set_xticks(range(1, len(METHOD_ORDER) + 1), [METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_ylabel(r"$\log_{10}(\mathrm{MSE}/\mathrm{MSE}_{Raw})$")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    figure_footer(fig)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.12, right=0.97)
    save_figure(fig, output_root, "fig04-relative-mse")


def plot_mse_by_batch(candidate_df: pd.DataFrame, model: str, output_root: Path, stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9), sharex=True)
    figure_header(fig, f"{MODEL_LABEL[model]}：MSE 随样本预算 B 的变化", "每个点为相应方法在 64 次重复后的均值向量误差；纵轴为对数尺度")
    for ax, stage in zip(axes, STAGE_ORDER):
        subset = candidate_df[(candidate_df["model"] == model) & (candidate_df["stage"] == stage)]
        for method in METHOD_ORDER:
            method_rows = subset[subset["method"] == method].sort_values("batch_size")
            ax.plot(
                method_rows["batch_size"],
                method_rows["mse"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=1.7,
                markersize=5.5,
                label=METHOD_LABEL[method],
            )
        ax.set_title(STAGE_LABEL[stage], fontsize=11, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256], ["32", "64", "128", "256"])
        ax.set_xlabel("样本预算 B")
        ax.grid(True, which="major")
    axes[0].set_ylabel("MSE（log）")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.015), ncol=7, fontsize=8.5)
    figure_footer(fig)
    fig.subplots_adjust(top=0.82, bottom=0.21, left=0.065, right=0.985, wspace=0.20)
    save_figure(fig, output_root, stem)


def plot_pearson(candidate_df: pd.DataFrame, output_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    figure_header(fig, "各估计器的 Pearson 排序保持度", "跨 24 个 cell×B 场景；纵轴聚焦观测区间，箱体表示中位数与四分位范围")
    values = [candidate_df.loc[candidate_df["method"] == method, "pearson"].to_numpy() for method in METHOD_ORDER]
    box = ax.boxplot(values, patch_artist=True, widths=0.58, showfliers=True, medianprops={"color": INK, "linewidth": 1.7})
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.30)
        patch.set_edgecolor(METHOD_COLORS[method])
    ax.set_xticks(range(1, len(METHOD_ORDER) + 1), [METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_ylabel("Pearson 相关系数")
    ax.set_ylim(max(-1.0, float(candidate_df["pearson"].min()) - 0.04), 1.01)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    figure_footer(fig)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.11, right=0.97)
    save_figure(fig, output_root, "fig07-pearson-distribution")


def plot_bias_variance(method_summary: pd.DataFrame, output_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    figure_header(fig, "估计器的偏差—重复方差位置", "每个点为 24 个 cell×B 场景的中位数；左下方向同时降低绝对偏差与跨重复方差")
    offsets = {
        "raw": (7, 6),
        "double": (9, 11),
        "u_m4": (9, 10),
        "u_m8": (9, 5),
        "u_m16": (9, 14),
        "u_m32": (9, -4),
    }
    for row in method_summary.itertuples(index=False):
        if row.method == "u_m2":
            continue
        ax.scatter(
            row.median_abs_signed_bias,
            row.median_between_rep_variance,
            s=115,
            marker=METHOD_MARKERS[row.method],
            color=METHOD_COLORS[row.method],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            "Double ≡ U-2" if row.method == "double" else row.method_label,
            (row.median_abs_signed_bias, row.median_between_rep_variance),
            xytext=offsets[row.method],
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=METHOD_COLORS[row.method],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|signed bias mean| 的中位数（log）")
    ax.set_ylabel("逐坐标跨重复方差均值的中位数（log）")
    ax.grid(True, which="major")
    figure_footer(fig)
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.14, right=0.97)
    save_figure(fig, output_root, "fig08-bias-variance")


def plot_batch_frequency(batch_frequency: pd.DataFrame, output_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    figure_header(fig, "每个 cell×估计器的最低 MSE 所对应样本预算", "共 42 个 cell×method 比较；柱高为该 B 成为最低 MSE 的次数")
    bars = ax.bar(
        batch_frequency["batch_size"].astype(str),
        batch_frequency["best_configuration_count"],
        color=[BATCH_COLORS[int(value)] for value in batch_frequency["batch_size"]],
        edgecolor="white",
        width=0.62,
    )
    for bar, count, share in zip(bars, batch_frequency["best_configuration_count"], batch_frequency["share"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.45, f"{int(count)}\n({share:.0%})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("样本预算 B")
    ax.set_ylabel("最低 MSE 次数")
    ax.set_ylim(0, max(batch_frequency["best_configuration_count"]) * 1.24)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    figure_footer(fig)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.10, right=0.97)
    save_figure(fig, output_root, "fig09-best-batch-frequency")


def plot_profiler_lines(profiler_df: pd.DataFrame, output_root: Path, metric: str, ylabel: str, title: str, subtitle: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), sharex=True)
    figure_header(fig, title, subtitle)
    stage_colors = {"initialization": BLUE, "early": ORANGE, "mid_late": PURPLE}
    stage_markers = {"initialization": "o", "early": "s", "mid_late": "^"}
    for ax, model in zip(axes, MODEL_ORDER):
        subset = profiler_df[profiler_df["model"] == model]
        for stage in STAGE_ORDER:
            rows = subset[subset["stage"] == stage].sort_values("batch_size")
            ax.plot(
                rows["batch_size"],
                rows[metric],
                color=stage_colors[stage],
                marker=stage_markers[stage],
                linewidth=2.1,
                markersize=6,
                label=STAGE_LABEL[stage],
            )
        ax.set_title(MODEL_LABEL[model], fontsize=11, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256], ["32", "64", "128", "256"])
        ax.set_xlabel("样本预算 B")
        ax.grid(True, which="major")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.02), ncol=3)
    figure_footer(fig)
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.085, right=0.98, wspace=0.18)
    save_figure(fig, output_root, stem)


def plot_accuracy_cost(best_by_cell_batch: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0))
    figure_header(fig, "每个 cell 的精度—时间权衡", "每个 B 仅保留该 cell×B 中 MSE 最低的方法；纵轴为 MSE 对数尺度")
    axes_flat = axes.ravel()
    for ax, cell in zip(axes_flat, CELL_ORDER):
        rows = best_by_cell_batch[best_by_cell_batch["cell_id"] == cell].sort_values("batch_size")
        for row in rows.itertuples(index=False):
            ax.scatter(
                row.wall_seconds_mean,
                row.mse,
                s=82,
                color=BATCH_COLORS[int(row.batch_size)],
                marker=METHOD_MARKERS[row.method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                f"B{int(row.batch_size)} · {METHOD_LABEL[row.method]}",
                (row.wall_seconds_mean, row.mse),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
            )
        ax.set_title(CELL_LABEL[cell], fontsize=9.8, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlabel("每次重复壁钟时间（秒）", fontsize=8.5)
        ax.set_ylabel("最低 MSE", fontsize=8.5)
        ax.grid(True, which="major")
    batch_handles = [Patch(facecolor=BATCH_COLORS[b], label=f"B={b}") for b in [32, 64, 128, 256]]
    fig.legend(batch_handles, [handle.get_label() for handle in batch_handles], loc="lower center", bbox_to_anchor=(0.52, 0.018), ncol=4, fontsize=8.5)
    figure_footer(fig)
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.075, right=0.985, hspace=0.34, wspace=0.25)
    save_figure(fig, output_root, "fig12-accuracy-cost")


def write_outputs(
    candidate_df: pd.DataFrame,
    profiler_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    output_root: Path,
) -> None:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(data_root / "candidate-metrics.csv", index=False, encoding="utf-8-sig")
    profiler_df.to_csv(data_root / "profiler-metrics.csv", index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(data_root / f"{name.replace('_', '-')}.csv", index=False, encoding="utf-8-sig")

    double_rows = candidate_df[candidate_df["method"] == "double"].sort_values(["cell_id", "batch_size"])
    u2_rows = candidate_df[candidate_df["method"] == "u_m2"].sort_values(["cell_id", "batch_size"])
    identity = {
        metric: float(np.max(np.abs(double_rows[metric].to_numpy() - u2_rows[metric].to_numpy())))
        for metric in ["mse", "mae", "pearson", "signed_bias_mean", "between_repetition_variance_mean"]
    }

    recommendations = tables["recommendations"]
    method_summary = tables["method_summary"]
    summary = {
        **metadata,
        "candidate_count": int(len(candidate_df)),
        "cell_count": int(candidate_df["cell_id"].nunique()),
        "method_count": int(candidate_df["method"].nunique()),
        "batch_size_count": int(candidate_df["batch_size"].nunique()),
        "successful_repetition_summary": {
            "minimum": int(candidate_df["successful_repetitions"].min()),
            "maximum": int(candidate_df["successful_repetitions"].max()),
            "rows_below_64": int((candidate_df["successful_repetitions"] < 64).sum()),
            "affected_cell_batches": sorted(
                {
                    f"{row.cell_id}|B={int(row.batch_size)}|n={int(row.successful_repetitions)}"
                    for row in candidate_df.itertuples(index=False)
                    if int(row.successful_repetitions) < 64
                }
            ),
        },
        "double_u2_identity_max_abs": identity,
        "method_summary": method_summary.to_dict(orient="records"),
        "cell_recommendations": recommendations[
            ["cell_id", "method", "batch_size", "mse", "mae", "pearson", "successful_repetitions"]
        ].to_dict(orient="records"),
    }
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary["analysis_payload_hash"] = hashlib.sha256(encoded).hexdigest()
    with (data_root / "analysis-summary.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    candidate_df, profiler_df, metadata = load_source_tables(input_root)
    candidate_df = add_derived_metrics(candidate_df)
    tables = build_summary_tables(candidate_df, profiler_df)
    write_outputs(candidate_df, profiler_df, tables, metadata, output_root)

    plot_coverage(candidate_df, output_root)
    plot_recommendation_map(tables["recommendations"], output_root)
    plot_method_rank(tables["method_summary"], output_root)
    plot_relative_mse(candidate_df, output_root)
    plot_mse_by_batch(candidate_df, "pythia-14m", output_root, "fig05-mse-by-batch-14m")
    plot_mse_by_batch(candidate_df, "pythia-31m-deduped", output_root, "fig06-mse-by-batch-31m")
    plot_pearson(candidate_df, output_root)
    plot_bias_variance(tables["method_summary"], output_root)
    plot_batch_frequency(tables["batch_frequency"], output_root)
    plot_profiler_lines(
        profiler_df,
        output_root,
        metric="wall_seconds_mean",
        ylabel="每次重复壁钟时间（秒）",
        title="样本预算与壁钟时间",
        subtitle="每个 B 的一次 repetition 同时计算 Raw、Double 与全部 U-M；时间主要随梯度样本预算增长",
        stem="fig10-runtime-scaling",
    )
    plot_profiler_lines(
        profiler_df,
        output_root,
        metric="peak_memory_gib",
        ylabel="峰值显存（GiB）",
        title="样本预算与峰值显存",
        subtitle="峰值为每个 cell×B 的最大观测值；按模型分面并聚焦各自纵轴范围",
        stem="fig11-memory-scaling",
    )
    plot_accuracy_cost(tables["best_by_cell_batch"], output_root)

    print(json.dumps({"status": "COMPLETE", "output_root": str(output_root), "figures": 12, "candidates": 168}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
