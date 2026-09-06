from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.models import (
    LinearSpectralOperator,
    ModernSpectralOperator,
    SpectralOperator,
    spectral_basis,
)
from coherencygraph_das.spectral import spectral_mixture_operator


def test_graph_operator_shapes_and_psd_decoder() -> None:
    model = SpectralOperator(
        features=20,
        hidden_dim=32,
        layers=2,
        dropout=0.0,
        bands=4,
        q_bins=33,
        lags=[3, 5, 8],
        graph=True,
    )
    prediction, probabilities = model(torch.randn(5, 8, 20))
    assert prediction.shape == (5, 8, 4, 3, 2)
    assert probabilities.shape == (5, 8, 4, 33)
    assert torch.allclose(probabilities.sum(-1), torch.ones(5, 8, 4), atol=1e-6)
    operator = spectral_mixture_operator(
        probabilities[0, 0, 0].detach().numpy(), np.arange(32)
    )
    assert np.linalg.eigvalsh(operator).min() > -1e-9


def test_modern_backbones_preserve_shape_and_spectral_simplex() -> None:
    for family in [
        "gatv2_psd", "transformer_psd", "graphgps_psd", "bissm_psd", "gat_bissm_psd",
        "bigru_psd", "conv1d_psd",
    ]:
        model = ModernSpectralOperator(
            features=20,
            hidden_dim=32,
            layers=2,
            dropout=0.0,
            bands=4,
            q_bins=33,
            lags=[3, 5, 8],
            family=family,
            heads=4,
        )
        prediction, probabilities = model(torch.randn(3, 8, 20))
        assert prediction.shape == (3, 8, 4, 3, 2)
        assert probabilities.shape == (3, 8, 4, 33)
        assert torch.isfinite(prediction).all()
        assert torch.all(probabilities >= 0)
        assert torch.allclose(probabilities.sum(-1), torch.ones(3, 8, 4), atol=1e-6)


def test_coordinate_parameterizations_have_identical_phase_basis() -> None:
    lags = [3, 5, 8, 13, 21, 34, 55, 89]
    cosine, sine, _, _ = spectral_basis(257, lags, mode="channel_lag")
    reference = torch.complex(cosine, sine)
    for mode in ["metres_nyquist", "physical_rad_per_m", "normalized_max_lag"]:
        cosine, sine, _, _ = spectral_basis(
            257, lags, mode=mode, channel_spacing_m=9.5714288, maximum_lag=89
        )
        assert torch.max(torch.abs(torch.complex(cosine, sine) - reference)) < 1e-5


def test_alias_free_grid_and_linear_decoder_constraints() -> None:
    lags = [3, 5, 8, 13, 21, 34, 55, 89]
    assert 257 > 2 * max(lags)
    model = LinearSpectralOperator(7, 4, 257, lags, coordinate_mode="channel_lag")
    prediction, probabilities = model(torch.randn(2, 8, 7))
    assert prediction.shape == (2, 8, 4, 8, 2)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]), atol=1e-6)
    positions = torch.arange(90, dtype=torch.float32)
    q = model.spatial_wavenumber
    weights = probabilities[0, 0, 0]
    phase = (positions[:, None] - positions[None, :])[:, :, None] * q
    matrix = torch.sum(weights[None, None, :] * torch.exp(1j * phase), dim=-1)
    assert torch.max(torch.abs(torch.diag(matrix) - 1.0)) < 1e-5
    assert torch.linalg.eigvalsh(matrix).min() > -1e-4
