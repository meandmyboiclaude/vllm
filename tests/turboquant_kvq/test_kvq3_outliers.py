# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVQ-3: per-value outlier side-channel — CPU codec + accounting tests.

Run:
  tests/turboquant_kvq$ ~/shared/needfit/lens-venv/bin/python test_kvq3_outliers.py

Validates exact (bit-exact fp16) recovery of the retained top-|v| elements,
the MSE improvement over plain nuqv (large on heavy-tailed value
distributions), and the honest slot-size growth of the outlier presets.
"""

import torch

from _codec_ref import (
    fp16_retain,
    nuqv_value_codec,
    outlier_select,
    outlier_value_codec,
)
from _tqload import TQ_PRESETS, TurboQuantConfig

HEAD_DIM = 256


def _mse(a, b):
    return ((a - b) ** 2).mean().item()


def _gaussian(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float32)


def _heavy_tailed(n, d, seed=1, df=3.0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g, dtype=torch.float32)
    chi2 = torch.distributions.Chi2(df).sample((n, d))
    return z / torch.sqrt(chi2 / df)


def test_preset_outlier_count_and_slot_growth():
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv_out1", HEAD_DIM)
    nuqv = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    # 1% of 256 = 2.56 -> round -> 3 outliers.
    assert cfg.n_value_outliers == 3
    assert cfg.value_outliers_enabled is True
    # Additive slot growth: 3 outliers * 3 bytes = 9 bytes on the value side.
    assert cfg.value_outlier_bytes == 9
    assert cfg.value_packed_size == nuqv.value_packed_size + 9
    assert cfg.slot_size == nuqv.slot_size + 9
    # Outlier region sits right after packed indices + (scale, zero).
    assert cfg.value_outlier_offset == nuqv.value_packed_size - 4 + 4


def test_outlier_offset_matches_kernel_layout():
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv_out1", HEAD_DIM)
    import math
    data_bytes = math.ceil(HEAD_DIM * cfg.value_quant_bits / 8)
    assert cfg.value_outlier_offset == data_bytes + 4


def test_outliers_bit_exact_recovery():
    x = _heavy_tailed(200, HEAD_DIM, seed=9)
    n = 3
    out = outlier_value_codec(x, bits=3, n=n)
    idx, val = outlier_select(x, n)
    rows = torch.arange(x.shape[0]).unsqueeze(-1)
    # Retained positions reconstruct exactly as their fp16 cast.
    assert torch.equal(out[rows, idx], fp16_retain(val))


def test_outlier_channel_saturates_beyond_fp16():
    # gh-54085 follow-up 2: the kernel clamps each outlier into the fp16 finite
    # range before storing it, so a magnitude above 65504 comes back saturated,
    # not as inf. fp16_retain (the KVQ-2 sink model, whose producer does not
    # clamp) still models the raw cast — the two channels differ on purpose.
    x = torch.zeros(1, HEAD_DIM, dtype=torch.float32)
    x[0, 1:] = torch.linspace(-1.0, 4.0, HEAD_DIM - 1)
    x[0, 0] = -125000.0
    out = outlier_value_codec(x, bits=3, n=1)
    assert torch.isfinite(out).all(), "outlier channel reconstructed inf"
    assert out[0, 0].item() == -65504.0
    assert not torch.isfinite(fp16_retain(x[0, 0])), (
        "fp16_retain must keep modelling the unclamped sink cast"
    )


def test_outliers_beat_nuqv_heavy_tailed():
    x = _heavy_tailed(4000, HEAD_DIM)
    mse_nuqv = _mse(nuqv_value_codec(x, 3), x)
    mse_out = _mse(outlier_value_codec(x, 3, 3), x)
    assert mse_out < mse_nuqv
    # Heavy tails: retaining 3 elements exactly should cut error substantially.
    assert mse_out < 0.5 * mse_nuqv


def test_outliers_beat_nuqv_gaussian():
    x = _gaussian(4000, HEAD_DIM)
    mse_nuqv = _mse(nuqv_value_codec(x, 3), x)
    mse_out = _mse(outlier_value_codec(x, 3, 3), x)
    assert mse_out < mse_nuqv


def test_outliers_never_worse_than_nuqv_elementwise_on_retained():
    # The retained elements' error can only go down (exact fp16 vs 3-bit).
    x = _heavy_tailed(500, HEAD_DIM, seed=11)
    idx, val = outlier_select(x, 3)
    rows = torch.arange(x.shape[0]).unsqueeze(-1)
    nuqv = nuqv_value_codec(x, 3)
    out = outlier_value_codec(x, 3, 3)
    err_nuqv = (nuqv[rows, idx] - val).abs()
    err_out = (out[rows, idx] - val).abs()
    assert torch.all(err_out <= err_nuqv + 1e-6)


def test_composed_out1_sink32_preset():
    cfg = TurboQuantConfig.from_cache_dtype(
        "turboquant_3bit_nuqv_out1_sink32", HEAD_DIM
    )
    assert cfg.value_nuq is True
    assert cfg.n_value_outliers == 3
    assert cfg.sink_tokens == 32
    # Composed slot size == nuqv + outlier bytes (sink is a side buffer).
    nuqv = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    assert cfg.slot_size == nuqv.slot_size + 9


def test_other_presets_have_no_outliers():
    for name in ("turboquant_3bit_nc", "turboquant_3bit_nuqv",
                 "turboquant_3bit_nuqv_sink32", "turboquant_4bit_nc"):
        cfg = TurboQuantConfig.from_cache_dtype(name, HEAD_DIM)
        assert cfg.n_value_outliers == 0
        assert cfg.value_outlier_bytes == 0


def test_presets_registered():
    assert "turboquant_3bit_nuqv_out1" in TQ_PRESETS
    assert "turboquant_3bit_nuqv_out1_sink32" in TQ_PRESETS


if __name__ == "__main__":
    from _run import run_module

    raise SystemExit(1 if run_module(globals()) else 0)
