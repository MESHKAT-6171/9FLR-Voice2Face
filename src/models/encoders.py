import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.models as models
from transformers import AutoModel


class MFCCSpeechEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, sample_rate: int = 16000):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=40, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )
        self.net = nn.Sequential(
            nn.Conv1d(40, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feat = self.mfcc(wav)               # (B, 40, T)
        out = self.net(feat).squeeze(-1)    # (B, 512)
        return F.normalize(self.proj(out), dim=-1)


class MelResNetSpeechEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, sample_rate: int = 16000):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, hop_length=160, n_mels=64
        )
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, embed_dim)
        self.backbone = backbone

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.log1p(self.melspec(wav)).unsqueeze(1)  # (B, 1, 64, T)
        out = self.backbone(spec)
        return F.normalize(out, dim=-1)


class TransformerSpeechEncoder(nn.Module):
    """Supports Wav2Vec 2.0 (facebook/wav2vec2-base) and HuBERT (facebook/hubert-base-ls960)."""
    def __init__(self, model_name: str = "facebook/wav2vec2-base", embed_dim: int = 512, freeze_layers: int = 8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.feature_extractor._freeze_parameters()
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
            for layer in self.encoder.encoder.layers[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        out = self.encoder(wav).last_hidden_state          # (B, T, H)
        mean_pool = out.mean(dim=1)
        std_pool = out.std(dim=1)
        pooled = torch.cat([mean_pool, std_pool], dim=-1)   # (B, 2H)
        return F.normalize(self.proj(pooled), dim=-1)


class Face9FLREncoder(nn.Module):
    """Encodes global face + 9 local facial regions during training."""
    def __init__(self, backbone_type: str = "resnet50", embed_dim: int = 512, use_9flr: bool = True):
        super().__init__()
        self.use_9flr = use_9flr
        if backbone_type == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone_type == "mobilenet_v2":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported face backbone: {backbone_type}")

        self.global_backbone = net
        self.global_proj = nn.Linear(feat_dim, embed_dim)

        if self.use_9flr:
            self.region_cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.region_proj = nn.Linear(64, embed_dim)

    def forward(self, face: torch.Tensor, regions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        g_feat = self.global_backbone(face)
        g_emb = F.normalize(self.global_proj(g_feat), dim=-1)  # (B, D)

        if not self.use_9flr or regions is None:
            return g_emb, None

        b, num_r, c, h, w = regions.shape
        r_flat = regions.view(b * num_r, c, h, w)
        r_feat = self.region_cnn(r_flat).view(b, num_r, 64)
        r_emb = F.normalize(self.region_proj(r_feat), dim=-1)  # (B, 9, D)
        return g_emb, r_emb