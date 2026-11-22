import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import Voice2FaceDataset
from src.losses import CompositeVoice2FaceLoss
from src.models.model_factory import MODEL_REGISTRY, build_model


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_single_run(
    model_name: str,
    split_name: str,
    processed_dir: Path,
    splits_dir: Path,
    results_dir: Path,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"{model_name}__{split_name}"
    ckpt_dir = results_dir / "checkpoints"
    log_dir = results_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / f"{run_id}_best.pth"
    log_csv_path = log_dir / f"{run_id}_log.csv"

    if best_ckpt_path.exists() and log_csv_path.exists():
        print(f"[SKIP] {run_id} already trained ({best_ckpt_path} exists).")
        return

    print(f"\n==========================================")
    print(f"Starting Run: {run_id} on {device}")
    print(f"==========================================")

    split_path = splits_dir / split_name
    train_ds = Voice2FaceDataset(split_path / "train.csv", processed_dir)
    val_ds = Voice2FaceDataset(split_path / "val.csv", processed_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(model_name).to(device)
    criterion = CompositeVoice2FaceLoss().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")

    with log_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_l1", "val_lpips", "val_cos"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                audio = batch["audio"].to(device, non_blocking=True)
                face = batch["face"].to(device, non_blocking=True)
                regions = batch["regions"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(audio, face, regions)
                    loss, _ = criterion(outputs, face)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))

            scheduler.step()

            # Validation
            model.eval()
            val_metrics = {"loss_total": [], "loss_l1": [], "loss_lpips": [], "loss_cos": []}
            with torch.no_grad():
                for batch in val_loader:
                    audio = batch["audio"].to(device, non_blocking=True)
                    face = batch["face"].to(device, non_blocking=True)
                    regions = batch["regions"].to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        outputs = model(audio, face, regions)
                        _, m = criterion(outputs, face)

                    for k in val_metrics:
                        val_metrics[k].append(m[k])

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_metrics["loss_total"]))
            avg_l1 = float(np.mean(val_metrics["loss_l1"]))
            avg_lpips = float(np.mean(val_metrics["loss_lpips"]))
            avg_cos = float(np.mean(val_metrics["loss_cos"]))

            writer.writerow({
                "epoch": epoch,
                "train_loss": round(avg_train, 5),
                "val_loss": round(avg_val, 5),
                "val_l1": round(avg_l1, 5),
                "val_lpips": round(avg_lpips, 5),
                "val_cos": round(avg_cos, 5),
            })
            f.flush()

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train: {avg_train:.4f} | "
                f"Val: {avg_val:.4f} (L1={avg_l1:.4f}, LPIPS={avg_lpips:.4f}, Cos={avg_cos:.4f})"
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(
                    {
                        "model_name": model_name,
                        "split_name": split_name,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "state_dict": model.state_dict(),
                    },
                    best_ckpt_path,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ALL", help="Model name or ALL")
    parser.add_argument("--split", type=str, default="ALL", help="split_60_30_10, split_70_20_10, or ALL")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    seed_everything(42)
    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "ALL" else [args.model]
    splits_to_run = ["split_60_30_10", "split_70_20_10"] if args.split == "ALL" else [args.split]

    for split_name in splits_to_run:
        for model_name in models_to_run:
            train_single_run(
                model_name=model_name,
                split_name=split_name,
                processed_dir=Path(args.processed_dir),
                splits_dir=Path(args.splits_dir),
                results_dir=Path(args.results_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )


if __name__ == "__main__":
    main()