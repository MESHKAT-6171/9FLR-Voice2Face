import random
from pathlib import Path

import pandas as pd
import yaml

# --- Load Configuration ---
CONFIG_PATH = Path("configs/config.yaml")
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED = int(CFG["project"]["seed"])
PROCESSED_DIR = Path(CFG["paths"]["processed_dir"])
SPLITS_DIR = Path(CFG["paths"]["splits_dir"])
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"


def calculate_folder_size_mb(folder: Path) -> float:
    total_bytes = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    return total_bytes / (1024 * 1024)


def verify_files_exist(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only rows where audio, face, and 9FLR region files all exist on disk."""
    valid_mask = [
        (PROCESSED_DIR / row.audio_path).exists()
        and (PROCESSED_DIR / row.face_path).exists()
        and (PROCESSED_DIR / row.regions_path).exists()
        for row in df.itertuples()
    ]
    return df[valid_mask].reset_index(drop=True)


def save_split_csvs(df: pd.DataFrame, split_name: str, train_vids: set, val_vids: set, test_vids: set):
    out_dir = SPLITS_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["video_id"].isin(train_vids)].reset_index(drop=True)
    val_df = df[df["video_id"].isin(val_vids)].reset_index(drop=True)
    test_df = df[df["video_id"].isin(test_vids)].reset_index(drop=True)

    # Sanity check: zero video_id overlap
    assert len(train_vids & val_vids) == 0, "Leakage between train and val!"
    assert len(train_vids & test_vids) == 0, "Leakage between train and test!"
    assert len(val_vids & test_vids) == 0, "Leakage between val and test!"

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    total = len(df)
    print(f"\n=== {split_name} (Saved to {out_dir}) ===")
    print(f"  Train : {len(train_vids):4d} videos | {len(train_df):4d} clips ({len(train_df)/total*100:.1f}%)")
    print(f"  Val   : {len(val_vids):4d} videos | {len(val_df):4d} clips ({len(val_df)/total*100:.1f}%)")
    print(f"  Test  : {len(test_vids):4d} videos | {len(test_df):4d} clips ({len(test_df)/total*100:.1f}%)")


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    raw_df = pd.read_csv(MANIFEST_PATH, dtype={"video_id": str, "sample_id": str})
    print(f"Loaded {len(raw_df)} rows from {MANIFEST_PATH}")

    df = verify_files_exist(raw_df)
    dropped = len(raw_df) - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} incomplete entries missing files on disk.")

    unique_videos = sorted(df["video_id"].unique().tolist())
    rng = random.Random(SEED)
    rng.shuffle(unique_videos)

    n_total = len(unique_videos)
    n_test = int(round(n_total * 0.10))
    test_vids = set(unique_videos[:n_test])
    remaining_vids = unique_videos[n_test:]  # 90% pool for Train + Val

    # 1. Split 60:30:10 (60/90 = 2/3 of remaining -> Train, 30/90 = 1/3 -> Val)
    n_train_60 = int(round(n_total * 0.60))
    train_vids_60 = set(remaining_vids[:n_train_60])
    val_vids_60 = set(remaining_vids[n_train_60:])
    save_split_csvs(df, "split_60_30_10", train_vids_60, val_vids_60, test_vids)

    # 2. Split 70:20:10 (70/90 of remaining -> Train, 20/90 -> Val)
    n_train_70 = int(round(n_total * 0.70))
    train_vids_70 = set(remaining_vids[:n_train_70])
    val_vids_70 = set(remaining_vids[n_train_70:])
    save_split_csvs(df, "split_70_20_10", train_vids_70, val_vids_70, test_vids)

    size_mb = calculate_folder_size_mb(PROCESSED_DIR)
    print(f"\nDataset Audit Complete:")
    print(f"  Verified Paired Samples : {len(df)}")
    print(f"  Unique Source Videos    : {n_total}")
    print(f"  Processed Folder Size   : {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")


if __name__ == "__main__":
    main()