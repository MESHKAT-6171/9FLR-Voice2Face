from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_training_curves(results_dir: Path, out_dir: Path):
    """Fig 4: Training & Validation Loss Curves across M1-M7 for both split ratios."""
    log_dir = results_dir / "logs"
    splits = ["split_60_30_10", "split_70_20_10"]
    titles = ["60:30:10 Split — Validation Loss Curves", "70:20:10 Split — Validation Loss Curves"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for ax, sp, title in zip(axes, splits, titles):
        log_files = sorted(log_dir.glob(f"*__{sp}_log.csv"))
        for lf in log_files:
            df = pd.read_csv(lf)
            label = lf.name.replace(f"__{sp}_log.csv", "")
            lw = 2.4 if "M7" in label else 1.3
            ls = "-" if "M7" in label else "--"
            ax.plot(df["epoch"], df["val_loss"], label=label, linewidth=lw, linestyle=ls)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Composite Validation Loss", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = out_dir / "fig4_training_validation_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_benchmark_comparison(results_dir: Path, out_dir: Path):
    """Fig 5: Side-by-Side Comparison of SSIM, LPIPS, Cosine Similarity & F1-Score across M1-M7."""
    bench_csv = results_dir / "benchmark_comparison.csv"
    if not bench_csv.exists():
        print("benchmark_comparison.csv not found yet.")
        return

    df = pd.read_csv(bench_csv)
    models = df["Model"].unique().tolist()
    short_names = [m.split("_")[0] for m in models]

    metrics = [
        ("SSIM_Up", "SSIM (Higher is Better ↑)"),
        ("LPIPS_Down", "LPIPS Perceptual Distance (Lower is Better ↓)"),
        ("Cosine_Sim_Up", "Identity Cosine Similarity (Higher is Better ↑)"),
        ("F1_Score", "Cross-Modal Verification F1-Score (Higher is Better ↑)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=300)
    x = np.arange(len(models))
    width = 0.36

    for ax, (col, title) in zip(axes.flat, metrics):
        v60 = [df[(df["Model"] == m) & (df["Split"] == "split_60_30_10")][col].values[0] for m in models]
        v70 = [df[(df["Model"] == m) & (df["Split"] == "split_70_20_10")][col].values[0] for m in models]

        b1 = ax.bar(x - width / 2, v60, width, label="60:30:10 Split", color="#457b9d")
        b2 = ax.bar(x + width / 2, v70, width, label="70:20:10 Split", color="#e76f51")

        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "fig5_model_benchmark_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    fig_dir = res_dir / "report_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(res_dir, fig_dir)
    plot_benchmark_comparison(res_dir, fig_dir)