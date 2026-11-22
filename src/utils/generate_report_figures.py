from pathlib import Path
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
OUT_DIR = Path("results/report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_NAMES = [
    "1. Forehead", "2. Left Eye", "3. Right Eye",
    "4. Nose Bridge", "5. Nose Tip", "6. Left Cheek",
    "7. Right Cheek", "8. Mouth", "9. Chin & Jaw"
]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_split_distributions():
    """Fig 1: Dual Split Distribution (60:30:10 vs 70:20:10) for Clips & Unique Videos."""
    splits = ["split_60_30_10", "split_70_20_10"]
    titles = ["60 : 30 : 10 Split Ratio", "70 : 20 : 10 Split Ratio"]
    subsets = ["train", "val", "test"]
    labels = ["Train", "Validation", "Test"]
    colors = ["#2b5c8f", "#4a90d9", "#e07a5f"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    for ax, sp, title in zip(axes, splits, titles):
        clip_counts = []
        vid_counts = []
        for sub in subsets:
            df = pd.read_csv(SPLITS_DIR / sp / f"{sub}.csv")
            clip_counts.append(len(df))
            vid_counts.append(df["video_id"].nunique())

        x = np.arange(len(labels))
        width = 0.36
        b1 = ax.bar(x - width / 2, clip_counts, width, label="Audio-Face Pairs (Clips)", color=colors)
        b2 = ax.bar(x + width / 2, vid_counts, width, label="Unique Speakers (Videos)", color="#81b29a")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, max(clip_counts) * 1.18)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 100, f"{h:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 100, f"{h:,}", ha="center", va="bottom", fontsize=8.5)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUT_DIR / "fig1_dataset_split_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_9flr_pipeline_showcase(sample_idx: int = 10):
    """Fig 2: Audio Waveform + Mel-Spectrogram + Aligned Face + 9 Facial Local Regions."""
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    row = manifest.iloc[sample_idx]

    wav, sr = sf.read(PROCESSED_DIR / row["audio_path"], dtype="float32")
    face_bgr = cv2.imread(str(PROCESSED_DIR / row["face_path"]))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    regions = np.load(PROCESSED_DIR / row["regions_path"])  # (9, 48, 48, 3)

    fig = plt.figure(figsize=(13, 5.5), dpi=300)
    gs = fig.add_gridspec(2, 6, width_ratios=[1.4, 1.4, 1.3, 0.85, 0.85, 0.85], wspace=0.35, hspace=0.4)

    # Top-left: Waveform
    ax_wav = fig.add_subplot(gs[0, 0:2])
    times = np.linspace(0, len(wav) / sr, len(wav))
    ax_wav.plot(times, wav, color="#1d3557", linewidth=0.6)
    ax_wav.set_title(f"(a) Standardized 16 kHz Speech Waveform ({row['sample_id']})", fontsize=10, fontweight="bold")
    ax_wav.set_xlabel("Time (s)", fontsize=8.5)
    ax_wav.set_ylabel("Amplitude", fontsize=8.5)
    ax_wav.set_xlim(0, times[-1])
    ax_wav.grid(alpha=0.3, linestyle="--")

    # Bottom-left: Log-Mel Spectrogram
    ax_mel = fig.add_subplot(gs[1, 0:2])
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=512, hop_length=160, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=ax_mel, cmap="magma")
    ax_mel.set_title("(b) Acoustic Log-Mel Spectrogram", fontsize=10, fontweight="bold")
    ax_mel.set_xlabel("Time (s)", fontsize=8.5)
    ax_mel.set_ylabel("Frequency (Hz)", fontsize=8.5)

    # Center: Aligned 128x128 Face
    ax_face = fig.add_subplot(gs[:, 2])
    ax_face.imshow(face_rgb)
    ax_face.set_title("(c) Aligned Face\n(128 × 128)", fontsize=10, fontweight="bold")
    ax_face.axis("off")

    # Right 3x3 Grid: 9 Facial Local Regions (9FLR)
    sub_gs = gs[:, 3:6].subgridspec(3, 3, wspace=0.15, hspace=0.35)
    for i in range(9):
        r, c = divmod(i, 3)
        ax_r = fig.add_subplot(sub_gs[r, c])
        ax_r.imshow(regions[i])
        ax_r.set_title(REGION_NAMES[i], fontsize=7.5, fontweight="bold", pad=3)
        ax_r.set_xticks([])
        ax_r.set_yticks([])

    fig.suptitle("(d) Extracted 9 Facial Local Regions (9FLR, 48 × 48 each)", x=0.77, y=0.96, fontsize=10, fontweight="bold")
    out_path = OUT_DIR / "fig2_9flr_multimodal_extraction.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_dataset_diversity_grid(num_samples: int = 16):
    """Fig 3: 4x4 Grid of Preprocessed Speaker Faces Illustrating Dataset Diversity."""
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv")
    unique_df = manifest.drop_duplicates(subset=["video_id"]).sample(n=num_samples, random_state=42)

    fig, axes = plt.subplots(4, 4, figsize=(7, 7.4), dpi=300)
    for ax, (_, row) in zip(axes.flat, unique_df.iterrows()):
        bgr = cv2.imread(str(PROCESSED_DIR / row["face_path"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"ID: {str(row['video_id'])[:8]}...", fontsize=7.5)
        ax.axis("off")

    fig.suptitle("Sample Aligned Speaker Faces from Curated Dataset (N = 2,060 Speakers)", fontsize=11, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "fig3_dataset_face_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_split_distributions()
    plot_9flr_pipeline_showcase(sample_idx=15)
    plot_dataset_diversity_grid(num_samples=16)
    print("\nAll Dataset & Methodology report figures generated in results/report_figures/")