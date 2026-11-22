import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import lpips
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import Voice2FaceDataset
from src.models.model_factory import MODEL_REGISTRY, build_model


def evaluate_checkpoint(
    ckpt_path: Path,
    processed_dir: Path,
    splits_dir: Path,
    vis_dir: Path,
    device: torch.device,
    lpips_fn,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt["model_name"]
    split_name = ckpt["split_name"]

    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds = Voice2FaceDataset(splits_dir / split_name / "test.csv", processed_dir)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    ssim_scores = []
    lpips_scores = []
    speech_embs = []
    face_embs = []
    saved_grid = False

    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            face = batch["face"].to(device)
            regions = batch["regions"].to(device)

            # Inference uses ONLY audio to predict face; face_encoder is only called to compute reference identity embedding
            outputs = model(audio, face, regions)
            pred_face = outputs["pred_face"].clamp(-1.0, 1.0)

            lp_val = lpips_fn(pred_face, face).squeeze().cpu().numpy()
            lpips_scores.extend(np.atleast_1d(lp_val).tolist())

            pred_np = ((pred_face.cpu().numpy() + 1.0) * 127.5).astype(np.uint8).transpose(0, 2, 3, 1)
            gt_np = ((face.cpu().numpy() + 1.0) * 127.5).astype(np.uint8).transpose(0, 2, 3, 1)

            for p_img, g_img in zip(pred_np, gt_np):
                s_val = ssim(g_img, p_img, channel_axis=2, data_range=255)
                ssim_scores.append(float(s_val))

            speech_embs.append(outputs["speech_emb"].cpu())
            face_embs.append(outputs["face_emb"].cpu())

            # Save side-by-side visual grid (Top: Ground Truth, Bottom: Predicted from Voice)
            if not saved_grid and len(pred_np) >= 6:
                gt_row = np.hstack(gt_np[:6])
                pr_row = np.hstack(pred_np[:6])
                grid = np.vstack([gt_row, pr_row])
                vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_dir / f"{model_name}__{split_name}_samples.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                saved_grid = True

    s_all = F.normalize(torch.cat(speech_embs, dim=0), dim=-1)
    f_all = F.normalize(torch.cat(face_embs, dim=0), dim=-1)

    # Paired Cosine Similarity
    paired_cos = F.cosine_similarity(s_all, f_all, dim=-1).numpy()

    # Cross-Modal Biometric Verification (Positive pairs vs Impostor pairs)
    n = len(s_all)
    sim_matrix = (s_all @ f_all.T).numpy()
    y_true = []
    y_pred = []
    threshold = float(np.median(sim_matrix))

    for i in range(n):
        # Positive pair
        y_true.append(1)
        y_pred.append(1 if sim_matrix[i, i] >= threshold else 0)
        # Impostor pair (shifted index)
        j = (i + 1) % n
        y_true.append(0)
        y_pred.append(1 if sim_matrix[i, j] >= threshold else 0)

    return {
        "Model": model_name,
        "Split": split_name,
        "Best_Epoch": ckpt["epoch"],
        "SSIM_Up": round(float(np.mean(ssim_scores)), 4),
        "LPIPS_Down": round(float(np.mean(lpips_scores)), 4),
        "Cosine_Sim_Up": round(float(np.mean(paired_cos)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    results_dir = Path(args.results_dir)
    ckpt_dir = results_dir / "checkpoints"
    vis_dir = results_dir / "visuals"

    rows = []
    for split_name in ["split_60_30_10", "split_70_20_10"]:
        for model_name in MODEL_REGISTRY.keys():
            ckpt_path = ckpt_dir / f"{model_name}__{split_name}_best.pth"
            if not ckpt_path.exists():
                print(f"[WARN] Missing checkpoint: {ckpt_path}")
                continue
            print(f"Evaluating {ckpt_path.name}...")
            row = evaluate_checkpoint(
                ckpt_path, Path(args.processed_dir), Path(args.splits_dir), vis_dir, device, lpips_fn
            )
            rows.append(row)

    if rows:
        out_csv = results_dir / "benchmark_comparison.csv"
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved Consolidated Benchmark Table to {out_csv}:\n")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()