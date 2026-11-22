import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    """
    Aligns speech embedding to global + 9FLR facial space.
    Works seamlessly during inference using speech_emb alone.
    """
    def __init__(self, embed_dim: int = 512, use_9flr: bool = True, num_regions: int = 9):
        super().__init__()
        self.use_9flr = use_9flr
        self.num_regions = num_regions

        if use_9flr:
            self.region_queries = nn.Parameter(torch.randn(1, num_regions, embed_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.out_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
        else:
            self.out_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )

    def forward(self, speech_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_9flr:
            return self.out_proj(speech_emb), None

        b, d = speech_emb.shape
        queries = self.region_queries.expand(b, -1, -1)                  # (B, 9, D)
        kv = speech_emb.unsqueeze(1)                                     # (B, 1, D)
        pred_regions, _ = self.cross_attn(query=queries, key=kv, value=kv)
        pred_regions = F.normalize(pred_regions, dim=-1)                 # (B, 9, D)

        region_summary = pred_regions.mean(dim=1)                        # (B, D)
        g = self.gate(torch.cat([speech_emb, region_summary], dim=-1))
        fused = self.out_proj(torch.cat([speech_emb * g, region_summary * (1.0 - g)], dim=-1))
        return fused, pred_regions


class StandardCNNDecoder(nn.Module):
    """Up-samples 512-d latent vector to 128x128 RGB face."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 512 * 4 * 4)
        self.net = nn.Sequential(
            self._block(512, 256),  # 8x8
            self._block(256, 128),  # 16x16
            self._block(128, 64),   # 32x32
            self._block(64, 32),    # 64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.net(x)


class UNetStyleDecoder(nn.Module):
    """Multi-scale residual U-Net decoder with skip refinement for sharper facial details."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 512 * 4 * 4)
        self.skip_proj_16 = nn.Linear(embed_dim, 128)
        self.skip_proj_32 = nn.Linear(embed_dim, 64)

        self.up1 = self._up_block(512, 256)          # 8x8
        self.up2 = self._up_block(256, 128)          # 16x16
        self.refine16 = self._res_block(128)
        self.up3 = self._up_block(128, 64)           # 32x32
        self.refine32 = self._res_block(64)
        self.up4 = self._up_block(64, 32)            # 64x64
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Tanh(),
        )

    @staticmethod
    def _up_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    @staticmethod
    def _res_block(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = x + self.skip_proj_16(z).unsqueeze(-1).unsqueeze(-1)
        x = x + self.refine16(x)
        x = self.up3(x)
        x = x + self.skip_proj_32(z).unsqueeze(-1).unsqueeze(-1)
        x = x + self.refine32(x)
        x = self.up4(x)
        return self.final(x)