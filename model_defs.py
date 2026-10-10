import torch
import torch.nn as nn


# ============================================================
# HR — revised ResTCN + squeeze/excitation + attention stats
# Input: [B, 3, 900]
# ============================================================


class SEBlock(nn.Module):
    """Squeeze/excitation block matching the frozen HR checkpoint."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResTCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        pad = 3 * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=7, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=7, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(c_out)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        z = self.conv1(x)
        z = self.norm1(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.conv2(z)
        z = self.norm2(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.se(z)
        return self.act(z + residual)


class AttentionStatsPool(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        a = torch.softmax(self.att(x), dim=-1)
        mean = torch.sum(a * x, dim=-1)
        second = torch.sum(a * (x ** 2), dim=-1)
        var = torch.clamp(second - mean ** 2, min=1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)


class HRResTCNAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=15, padding=7),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResTCNBlock(64, 64, dilation=1),
            ResTCNBlock(64, 96, dilation=2),
            ResTCNBlock(96, 128, dilation=4),
            ResTCNBlock(128, 128, dilation=8),
            ResTCNBlock(128, 128, dilation=16),
        )
        self.pool = AttentionStatsPool(128)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        z = self.stem(x)
        z = self.blocks(z)
        z = self.pool(z)
        return self.head(z).squeeze(1)


# ============================================================
# RR — frozen RR-v2 recording-level model, seed 42
# Input: [N_windows, 9, 900]
# State-dict naming matches best_rr_model.pt exactly:
# encoder.stem.*, encoder.body.*, head.*
# ============================================================


class RRResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = 3 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=7,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=7,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class RRWindowEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Exact checkpoint key layout:
        # stem.0 = Conv 9->32
        # stem.1 = BN32
        # stem.3 = Conv 32->32
        # stem.4 = BN32
        # stem.6 = Conv 32->64
        # stem.7 = BN64
        self.stem = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Exact checkpoint key layout:
        # body.0  = residual 64, d=1
        # body.1  = residual 64, d=2
        # body.2  = max-pool
        # body.3  = Conv 64->128
        # body.4  = BN128
        # body.5  = ReLU
        # body.6  = residual 128, d=2
        # body.7  = residual 128, d=4
        # body.8  = max-pool
        # body.9  = residual 128, d=8
        # body.10 = residual 128, d=16
        self.body = nn.Sequential(
            RRResidualBlock(64, dilation=1),
            RRResidualBlock(64, dilation=2),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            RRResidualBlock(128, dilation=2),
            RRResidualBlock(128, dilation=4),
            nn.MaxPool1d(2),
            RRResidualBlock(128, dilation=8),
            RRResidualBlock(128, dilation=16),
        )

    def forward(self, x):
        z = self.stem(x)
        z = self.body(z)
        # Per-window representation = global mean + global max => 256-D
        return torch.cat([z.mean(dim=-1), z.amax(dim=-1)], dim=1)


class RRRecordingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RRWindowEncoder()
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x contains all valid 30-s windows for one uploaded recording.
        w = self.encoder(x)  # [Nw, 256]
        rec = torch.cat([w.mean(dim=0), w.amax(dim=0)], dim=0).unsqueeze(0)  # [1,512]
        return self.head(rec).squeeze(1)


# ============================================================
# BP — revised Hybrid BP-v2 from Script 17
# RGB branch input: [N_windows, 3, 900]
# Robust session features: [1, 96]
# ============================================================


class BPCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x).mean(dim=-1)


class HybridBPNet(nn.Module):
    def __init__(self, n_features: int = 96):
        super().__init__()
        self.cnn = BPCNNEncoder()
        self.feature_branch = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.fusion = nn.Sequential(
            nn.Linear(320, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.sbp_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, rgb_windows, robust_features):
        w = self.cnn(rgb_windows)
        cnn_session = torch.cat([w.mean(dim=0), w.amax(dim=0)], dim=0).unsqueeze(0)
        handcrafted = self.feature_branch(robust_features)
        fused = self.fusion(torch.cat([cnn_session, handcrafted], dim=1))
        sbp = self.sbp_head(fused).squeeze(1)
        dbp = self.dbp_head(fused).squeeze(1)
        return torch.stack([sbp, dbp], dim=1)
