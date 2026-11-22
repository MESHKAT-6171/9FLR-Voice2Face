import torch
import torch.nn as nn
from src.models.encoders import (
    Face9FLREncoder,
    MFCCSpeechEncoder,
    MelResNetSpeechEncoder,
    TransformerSpeechEncoder,
)
from src.models.fusion_decoders import (
    FeatureFusionModule,
    StandardCNNDecoder,
    UNetStyleDecoder,
)

MODEL_REGISTRY = {
    "M1_mfcc_cnn": {
        "speech": "mfcc",
        "face": "resnet50",
        "use_9flr": False,
        "decoder": "cnn",
    },
    "M2_melspec_unet": {
        "speech": "melspec",
        "face": "resnet50",
        "use_9flr": False,
        "decoder": "unet",
    },
    "M3_w2v_global_cnn": {
        "speech": "wav2vec2",
        "face": "resnet50",
        "use_9flr": False,
        "decoder": "cnn",
    },
    "M4_w2v_global_unet": {
        "speech": "wav2vec2",
        "face": "resnet50",
        "use_9flr": False,
        "decoder": "unet",
    },
    "M5_w2v_9flr_mobilenet": {
        "speech": "wav2vec2",
        "face": "mobilenet_v2",
        "use_9flr": True,
        "decoder": "unet",
    },
    "M6_hubert_9flr_unet": {
        "speech": "hubert",
        "face": "resnet50",
        "use_9flr": True,
        "decoder": "unet",
    },
    "M7_proposed_9flr": {
        "speech": "wav2vec2",
        "face": "resnet50",
        "use_9flr": True,
        "decoder": "unet",
    },
}


class Voice2FaceFramework(nn.Module):
    def __init__(self, spec: dict, embed_dim: int = 512):
        super().__init__()
        self.use_9flr = spec["use_9flr"]

        # 1. Speech Encoder
        if spec["speech"] == "mfcc":
            self.speech_encoder = MFCCSpeechEncoder(embed_dim=embed_dim)
        elif spec["speech"] == "melspec":
            self.speech_encoder = MelResNetSpeechEncoder(embed_dim=embed_dim)
        elif spec["speech"] == "wav2vec2":
            self.speech_encoder = TransformerSpeechEncoder("facebook/wav2vec2-base", embed_dim=embed_dim)
        elif spec["speech"] == "hubert":
            self.speech_encoder = TransformerSpeechEncoder("facebook/hubert-base-ls960", embed_dim=embed_dim)
        else:
            raise ValueError(f"Unknown speech encoder: {spec['speech']}")

        # 2. Face Encoder (Training Guidance)
        self.face_encoder = Face9FLREncoder(
            backbone_type=spec["face"], embed_dim=embed_dim, use_9flr=self.use_9flr
        )

        # 3. Feature Fusion
        self.fusion = FeatureFusionModule(embed_dim=embed_dim, use_9flr=self.use_9flr)

        # 4. Reconstruction Decoder
        if spec["decoder"] == "cnn":
            self.decoder = StandardCNNDecoder(embed_dim=embed_dim)
        elif spec["decoder"] == "unet":
            self.decoder = UNetStyleDecoder(embed_dim=embed_dim)
        else:
            raise ValueError(f"Unknown decoder: {spec['decoder']}")

    def forward(self, audio: torch.Tensor, face: torch.Tensor | None = None, regions: torch.Tensor | None = None) -> dict:
        speech_emb = self.speech_encoder(audio)
        fused_z, pred_region_embs = self.fusion(speech_emb)
        pred_face = self.decoder(fused_z)

        out = {
            "pred_face": pred_face,
            "speech_emb": speech_emb,
            "pred_region_embs": pred_region_embs,
        }

        if face is not None:
            face_emb, target_region_embs = self.face_encoder(face, regions)
            out["face_emb"] = face_emb
            out["target_region_embs"] = target_region_embs

        return out


def build_model(model_name: str, embed_dim: int = 512) -> Voice2FaceFramework:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return Voice2FaceFramework(MODEL_REGISTRY[model_name], embed_dim=embed_dim)