# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVQ-1: non-uniform (Lloyd-Max) value quantization — CPU codec tests.

Run:
  tests/turboquant_kvq$ \
    ~/shared/needfit/lens-venv/bin/python -m pytest test_kvq1_nuqv.py -v

Validates that the ``turboquant_3bit_nuqv`` value codec lowers reconstruction
MSE vs the existing uniform min/max codec at identical 3-bit storage and
identical slot size (capacity unchanged), on both Gaussian and heavy-tailed
value distributions.
"""

import torch

from _codec_ref import (
    nuqv_encode,
    nuqv_value_codec,
    uniform_value_codec,
)
from _tqload import (
    TQ_PRESETS,
    TurboQuantConfig,
    get_value_codebook,
)

HEAD_DIM = 256


def _mse(a, b):
    return ((a - b) ** 2).mean().item()


def _gaussian(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float32)


def _heavy_tailed(n, d, seed=1, df=3.0):
    g = torch.Generator().manual_seed(seed)
    # Student-t via normal / sqrt(chi2/df); heavy tails stress the value codec.
    z = torch.randn(n, d, generator=g, dtype=torch.float32)
    chi2 = torch.distributions.Chi2(df).sample((n, d))
    return z / torch.sqrt(chi2 / df)


def test_value_codebook_shape_and_sorted():
    cent, mid = get_value_codebook(3)
    assert cent.numel() == 8
    assert mid.numel() == 7
    assert torch.all(cent[1:] > cent[:-1]), "centroids must be strictly sorted"
    assert torch.all(mid[1:] > mid[:-1]), "midpoints must be strictly sorted"
    # N(0,1) Lloyd-Max codebook is (approximately) zero-mean and symmetric.
    assert abs(cent.mean().item()) < 1e-3


def test_nuqv_beats_uniform_gaussian():
    x = _gaussian(4000, HEAD_DIM)
    mse_u = _mse(uniform_value_codec(x, 3), x)
    mse_n = _mse(nuqv_value_codec(x, 3), x)
    assert mse_n < mse_u, f"nuqv {mse_n:.5f} should beat uniform {mse_u:.5f}"
    # Expect a substantial gain on matched Gaussian data.
    assert mse_n < 0.8 * mse_u


def test_nuqv_beats_uniform_heavy_tailed():
    x = _heavy_tailed(4000, HEAD_DIM)
    mse_u = _mse(uniform_value_codec(x, 3), x)
    mse_n = _mse(nuqv_value_codec(x, 3), x)
    assert mse_n < mse_u, f"nuqv {mse_n:.5f} should beat uniform {mse_u:.5f}"


def test_nuqv_indices_in_range():
    x = _gaussian(500, HEAD_DIM, seed=7)
    idx, std, mean = nuqv_encode(x, 3)
    assert idx.min().item() >= 0
    assert idx.max().item() <= 7
    assert idx.shape == x.shape
    assert std.shape == (500, 1) and mean.shape == (500, 1)


def test_nuqv_recovers_constant_vector_exactly():
    # A constant vector has std 0 -> clamped; every element maps to the
    # centroid nearest 0, offset by mean. Reconstruction error is bounded by
    # the codebook's central spacing times the (tiny) clamp scale.
    x = torch.full((1, HEAD_DIM), 3.14159, dtype=torch.float32)
    out = nuqv_value_codec(x, 3)
    assert torch.allclose(out, x, atol=1e-2)


def test_codecs_saturate_beyond_fp16_like_the_kernel():
    # gh-54085 follow-up 2: the kernels clamp every fp16 side channel into
    # [-65504, 65504] before storing it, so the CPU reference has to as well —
    # otherwise the two disagree exactly where the bug used to live (an
    # unclamped cast gives inf, and one inf poisons the whole vector).
    x = torch.zeros(2, HEAD_DIM, dtype=torch.float32)
    x[0, 1:] = torch.linspace(-1.0, 4.0, HEAD_DIM - 1)
    x[0, 0] = -125000.0  # one end beyond fp16 range
    x[1, 1:] = torch.linspace(-1.0, 4.0, HEAD_DIM - 1)
    x[1, 0] = -125000.0
    x[1, 1] = 125000.0  # both ends beyond fp16 range

    out_u = uniform_value_codec(x, 3)
    assert torch.isfinite(out_u).all(), "uniform reference reconstructed inf"

    _, std, mean = nuqv_encode(x, 3)
    assert torch.isfinite(std).all() and torch.isfinite(mean).all()
    assert std.max().item() <= 65504.0
    assert mean.abs().max().item() <= 65504.0


def test_preset_slot_size_unchanged_vs_3bit_nc():
    nuqv = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    nc = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nc", HEAD_DIM)
    assert nuqv.value_nuq is True
    assert nc.value_nuq is False
    assert nuqv.slot_size == nc.slot_size, "slot size must be unchanged (KVQ-1)"
    assert nuqv.slot_size_aligned == nc.slot_size_aligned
    assert nuqv.value_packed_size == nc.value_packed_size
    assert nuqv.key_packed_size == nc.key_packed_size


def test_existing_presets_behavior_unchanged():
    # Additive-only guarantee: existing presets keep value_nuq == False.
    for name in ("turboquant_k8v4", "turboquant_4bit_nc",
                 "turboquant_k3v4_nc", "turboquant_3bit_nc"):
        cfg = TurboQuantConfig.from_cache_dtype(name, HEAD_DIM)
        assert cfg.value_nuq is False, f"{name} must remain uniform-value"


def test_nuqv_preset_registered():
    assert "turboquant_3bit_nuqv" in TQ_PRESETS
    assert TQ_PRESETS["turboquant_3bit_nuqv"]["value_nuq"] is True


if __name__ == "__main__":
    from _run import run_module

    raise SystemExit(1 if run_module(globals()) else 0)
