from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def spectral_coordinates(
    q_bins: int,
    lags: list[int] | Tensor,
    mode: str = "legacy_channel_lag",
    channel_spacing_m: float = 9.5714288,
    maximum_lag: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return dimensionally paired separation and wavenumber coordinates.

    ``legacy_channel_lag`` reproduces the original implementation exactly.  The
    remaining modes use FFT-centred integer harmonics.  Channel, metre and
    normalized coordinates are changes of units and therefore produce the same
    phase matrix when their wavenumbers are scaled consistently.
    """

    if q_bins < 3:
        raise ValueError("q_bins must be at least three")
    lag = torch.as_tensor(lags, dtype=torch.float64)
    if mode == "legacy_channel_lag":
        q = torch.linspace(-math.pi, math.pi, q_bins + 1, dtype=torch.float64)[:-1]
        return lag, q

    harmonics = torch.fft.fftfreq(q_bins, d=1.0, dtype=torch.float64) * (2.0 * math.pi)
    harmonics = torch.sort(harmonics).values
    if mode == "channel_lag":
        return lag, harmonics
    if mode in {"metres_nyquist", "physical_rad_per_m"}:
        spacing = float(channel_spacing_m)
        if spacing <= 0:
            raise ValueError("channel_spacing_m must be positive")
        return lag * spacing, harmonics / spacing
    if mode == "normalized_max_lag":
        scale = float(maximum_lag if maximum_lag is not None else torch.max(lag).item())
        if scale <= 0:
            raise ValueError("maximum_lag must be positive")
        return lag / scale, harmonics * scale
    raise ValueError(f"unknown spectral coordinate mode: {mode}")


def spectral_basis(
    q_bins: int,
    lags: list[int] | Tensor,
    mode: str = "legacy_channel_lag",
    channel_spacing_m: float = 9.5714288,
    maximum_lag: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    separation, q = spectral_coordinates(
        q_bins,
        lags,
        mode=mode,
        channel_spacing_m=channel_spacing_m,
        maximum_lag=maximum_lag,
    )
    phase = separation[:, None] * q[None, :]
    return torch.cos(phase).to(torch.float32), torch.sin(phase).to(torch.float32), separation, q


class ChainGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        left = torch.cat([values[:, :1], values[:, :-1]], dim=1)
        right = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
        neighbours = (left + values + right) / 3.0
        return values + self.update(torch.cat([values, neighbours], dim=-1))


class GraphBackbone(nn.Module):
    def __init__(self, features: int, hidden_dim: int, layers: int, dropout: float, graph: bool = True) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(features, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(
            ChainGraphLayer(hidden_dim, dropout) if graph else nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
            )
            for _ in range(layers)
        )
        self.graph = graph

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.input(features)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class GraphAutoencoder(nn.Module):
    def __init__(self, features: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.backbone = GraphBackbone(features, hidden_dim, layers, dropout, graph=True)
        self.reconstruct = nn.Linear(hidden_dim, features)

    def forward(self, features: Tensor) -> Tensor:
        return self.reconstruct(self.backbone(features))


class SpectralOperator(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        bands: int,
        q_bins: int,
        lags: list[int],
        graph: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = GraphBackbone(features, hidden_dim, layers, dropout, graph=graph)
        self.logits = nn.Linear(hidden_dim, bands * q_bins)
        self.bands = bands
        self.q_bins = q_bins
        q = torch.linspace(-math.pi, math.pi, q_bins + 1)[:-1]
        lag = torch.tensor(lags, dtype=torch.float32)
        self.register_buffer("cosine", torch.cos(lag[:, None] * q[None, :]))
        self.register_buffer("sine", torch.sin(lag[:, None] * q[None, :]))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(features)
        logits = self.logits(hidden).reshape(*hidden.shape[:2], self.bands, self.q_bins)
        probabilities = torch.softmax(logits, dim=-1)
        real = torch.einsum("nxbq,lq->nxbl", probabilities, self.cosine)
        imag = torch.einsum("nxbq,lq->nxbl", probabilities, self.sine)
        return torch.stack([real, imag], dim=-1), probabilities


class UnconstrainedOperator(nn.Module):
    def __init__(self, features: int, hidden_dim: int, layers: int, dropout: float, bands: int, lags: int) -> None:
        super().__init__()
        self.backbone = GraphBackbone(features, hidden_dim, layers, dropout, graph=True)
        self.output = nn.Linear(hidden_dim, bands * lags * 2)
        self.bands = bands
        self.lags = lags

    def forward(self, features: Tensor) -> tuple[Tensor, None]:
        hidden = self.backbone(features)
        output = self.output(hidden).reshape(*hidden.shape[:2], self.bands, self.lags, 2)
        return torch.tanh(output), None


class DynamicChainAttention(nn.Module):
    """GATv2-style dynamic attention restricted to the fibre-chain neighbourhood."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attention)

    @staticmethod
    def _neighbours(values: Tensor) -> Tensor:
        left = torch.cat([values[:, :1], values[:, :-1]], dim=1)
        right = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
        return torch.stack([left, values, right], dim=2)

    def forward(self, values: Tensor) -> Tensor:
        n, blocks, _ = values.shape
        q = self.query(values).reshape(n, blocks, self.heads, self.head_dim)
        k = self._neighbours(self.key(values)).reshape(
            n, blocks, 3, self.heads, self.head_dim
        )
        v = self._neighbours(self.value(values)).reshape(
            n, blocks, 3, self.heads, self.head_dim
        )
        scores = torch.einsum(
            "nbshd,hd->nbsh",
            F.leaky_relu(q[:, :, None] + k, negative_slope=0.2),
            self.attention,
        ) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=2)
        mixed = torch.einsum("nbsh,nbshd->nbhd", self.dropout(weights), v)
        return self.output(mixed.reshape(n, blocks, -1))


class GATv2Block(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = DynamicChainAttention(hidden_dim, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm1(values))
        return values + self.ffn(self.norm2(values))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        values = values + attended
        return values + self.ffn(self.norm2(values))


class GPSBlock(nn.Module):
    """GraphGPS-style parallel local and global attention with learned fusion."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.local = DynamicChainAttention(hidden_dim, heads, dropout)
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm(values)
        local = self.local(normalized)
        global_values, _ = self.global_attention(
            normalized, normalized, normalized, need_weights=False
        )
        gate = torch.sigmoid(self.gate(torch.cat([local, global_values], dim=-1)))
        values = values + gate * local + (1.0 - gate) * global_values
        return values + self.ffn(self.norm2(values))


class BidirectionalGatedStateSpaceBlock(nn.Module):
    """Compact input-gated state-space recurrence for the short fibre sequence."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.forward_gate = nn.Linear(hidden_dim, hidden_dim)
        self.forward_input = nn.Linear(hidden_dim, hidden_dim)
        self.backward_gate = nn.Linear(hidden_dim, hidden_dim)
        self.backward_input = nn.Linear(hidden_dim, hidden_dim)
        self.mix = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _scan(values: Tensor, gate: nn.Linear, candidate: nn.Linear, reverse: bool) -> Tensor:
        indices = range(values.shape[1] - 1, -1, -1) if reverse else range(values.shape[1])
        state = torch.zeros_like(values[:, 0])
        outputs: list[Tensor] = []
        for index in indices:
            alpha = torch.sigmoid(gate(values[:, index]))
            proposal = torch.tanh(candidate(values[:, index]))
            state = alpha * state + (1.0 - alpha) * proposal
            outputs.append(state)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm(values)
        forward = self._scan(normalized, self.forward_gate, self.forward_input, False)
        backward = self._scan(normalized, self.backward_gate, self.backward_input, True)
        return values + self.mix(torch.cat([forward, backward], dim=-1))


class GatedAttentionStateSpaceBlock(nn.Module):
    """Parallel local dynamic attention and ordered global state propagation."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.local = DynamicChainAttention(hidden_dim, heads, dropout)
        self.state = BidirectionalGatedStateSpaceBlock(hidden_dim, dropout)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm(values)
        local = self.local(normalized)
        ordered = self.state(normalized) - normalized
        gate = torch.sigmoid(self.gate(torch.cat([local, ordered], dim=-1)))
        values = values + gate * local + (1.0 - gate) * ordered
        return values + self.ffn(self.norm2(values))


class BidirectionalGRUBlock(nn.Module):
    """Small ordered-sequence baseline with no graph-specific message passing."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for the bidirectional GRU")
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        recurrent, _ = self.gru(self.norm(values))
        values = values + self.dropout(recurrent)
        return values + self.ffn(self.norm2(values))


class ConvolutionalSequenceBlock(nn.Module):
    """Residual one-dimensional convolution baseline for the eight fibre blocks."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm(values).transpose(1, 2)
        convolved = self.pointwise(F.gelu(self.depthwise(normalized))).transpose(1, 2)
        values = values + self.dropout(convolved)
        return values + self.ffn(self.norm2(values))


class ModernBackbone(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        family: str,
        heads: int = 4,
        maximum_blocks: int = 16,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(features, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.position = nn.Parameter(torch.zeros(1, maximum_blocks, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        factories = {
            "gatv2_psd": lambda: GATv2Block(hidden_dim, heads, dropout),
            "transformer_psd": lambda: TransformerBlock(hidden_dim, heads, dropout),
            "graphgps_psd": lambda: GPSBlock(hidden_dim, heads, dropout),
            "bissm_psd": lambda: BidirectionalGatedStateSpaceBlock(hidden_dim, dropout),
            "gat_bissm_psd": lambda: GatedAttentionStateSpaceBlock(hidden_dim, heads, dropout),
            "bigru_psd": lambda: BidirectionalGRUBlock(hidden_dim, dropout),
            "conv1d_psd": lambda: ConvolutionalSequenceBlock(hidden_dim, dropout),
        }
        if family not in factories:
            raise ValueError(f"unknown modern backbone: {family}")
        self.layers = nn.ModuleList(factories[family]() for _ in range(layers))

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.input(features)
        hidden = hidden + self.position[:, : hidden.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class ModernSpectralOperator(nn.Module):
    """Recent sequence/graph backbones with the same PSD-preserving decoder."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        bands: int,
        q_bins: int,
        lags: list[int],
        family: str,
        heads: int = 4,
        coordinate_mode: str = "legacy_channel_lag",
        channel_spacing_m: float = 9.5714288,
        maximum_lag: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = ModernBackbone(
            features, hidden_dim, layers, dropout, family=family, heads=heads
        )
        self.logits = nn.Linear(hidden_dim, bands * q_bins)
        self.bands = bands
        self.q_bins = q_bins
        cosine, sine, separation, q = spectral_basis(
            q_bins,
            lags,
            mode=coordinate_mode,
            channel_spacing_m=channel_spacing_m,
            maximum_lag=maximum_lag,
        )
        self.coordinate_mode = coordinate_mode
        self.register_buffer("cosine", cosine)
        self.register_buffer("sine", sine)
        self.register_buffer("model_separation", separation)
        self.register_buffer("spatial_wavenumber", q)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(features)
        logits = self.logits(hidden).reshape(*hidden.shape[:2], self.bands, self.q_bins)
        probabilities = torch.softmax(logits, dim=-1)
        real = torch.einsum("nxbq,lq->nxbl", probabilities, self.cosine)
        imag = torch.einsum("nxbq,lq->nxbl", probabilities, self.sine)
        return torch.stack([real, imag], dim=-1), probabilities


class LinearSpectralOperator(nn.Module):
    """Linear logits coupled to the same nonnegative spectral decoder."""

    def __init__(
        self,
        features: int,
        bands: int,
        q_bins: int,
        lags: list[int],
        coordinate_mode: str = "channel_lag",
        channel_spacing_m: float = 9.5714288,
        maximum_lag: int | None = None,
    ) -> None:
        super().__init__()
        self.logits = nn.Linear(features, bands * q_bins)
        self.bands = bands
        self.q_bins = q_bins
        cosine, sine, separation, q = spectral_basis(
            q_bins,
            lags,
            mode=coordinate_mode,
            channel_spacing_m=channel_spacing_m,
            maximum_lag=maximum_lag,
        )
        self.coordinate_mode = coordinate_mode
        self.register_buffer("cosine", cosine)
        self.register_buffer("sine", sine)
        self.register_buffer("model_separation", separation)
        self.register_buffer("spatial_wavenumber", q)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.logits(features).reshape(*features.shape[:2], self.bands, self.q_bins)
        probabilities = torch.softmax(logits, dim=-1)
        real = torch.einsum("nxbq,lq->nxbl", probabilities, self.cosine)
        imag = torch.einsum("nxbq,lq->nxbl", probabilities, self.sine)
        return torch.stack([real, imag], dim=-1), probabilities


def masked_reconstruction_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return F.mse_loss(prediction, target)
    return ((prediction - target) ** 2)[mask].mean()
