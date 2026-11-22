import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeVoice2FaceLoss(nn.Module):
    def __init__(self, w_l1: float = 1.0, w_lpips: float = 0.5, w_cos: float = 0.5):
        super().__init__()
        self.w_l1 = w_l1
        self.w_lpips = w_lpips
        self.w_cos = w_cos
        self.lpips_fn = lpips.LPIPS(net="alex")
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def forward(self, outputs: dict, target_face: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pred_face = outputs["pred_face"]

        # 1. Pixel-level L1 reconstruction loss
        loss_l1 = F.l1_loss(pred_face, target_face)

        # 2. Perceptual LPIPS loss (expects [-1, 1])
        loss_perc = self.lpips_fn(pred_face, target_face).mean()

        # 3. Cross-modal Cosine Identity Alignment loss
        speech_emb = outputs["speech_emb"]
        face_emb = outputs["face_emb"]
        loss_cos = (1.0 - F.cosine_similarity(speech_emb, face_emb, dim=-1)).mean()

        # 4. 9FLR Local Region Alignment loss (when enabled)
        loss_region = torch.tensor(0.0, device=pred_face.device)
        if outputs.get("pred_region_embs") is not None and outputs.get("target_region_embs") is not None:
            loss_region = (
                1.0 - F.cosine_similarity(outputs["pred_region_embs"], outputs["target_region_embs"], dim=-1)
            ).mean()

        total = (
            self.w_l1 * loss_l1
            + self.w_lpips * loss_perc
            + self.w_cos * (loss_cos + 0.5 * loss_region)
        )
        metrics = {
            "loss_total": float(total.item()),
            "loss_l1": float(loss_l1.item()),
            "loss_lpips": float(loss_perc.item()),
            "loss_cos": float((loss_cos + 0.5 * loss_region).item()),
        }
        return total, metrics