# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVQ-4: per-layer bit allocation — resolver + grouping-invariant tests.

Run:
  tests/turboquant_kvq$ ~/shared/needfit/lens-venv/bin/python test_kvq4_layerbits.py

Validates the VLLM_TQ_LAYER_BITS resolver (default = uniform, per-layer
override, validation), and the slot-size relationships that determine KV-cache
grouping: presets that keep the same slot size (nc / nuqv) can be mixed within
a single group, while size-changing presets (out1) force separate groups.
"""

import os

from _tqload import (
    TQ_LAYER_BITS_ENV,
    TurboQuantConfig,
    get_tq_layer_bits_map,
    resolve_tq_layer_preset,
)

HEAD_DIM = 256


def _set_env(val):
    if val is None:
        os.environ.pop(TQ_LAYER_BITS_ENV, None)
    else:
        os.environ[TQ_LAYER_BITS_ENV] = val


def _expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_default_is_uniform_no_env():
    _set_env(None)
    assert get_tq_layer_bits_map() == {}
    for idx in range(16):
        assert (
            resolve_tq_layer_preset(idx, "turboquant_3bit_nuqv")
            == "turboquant_3bit_nuqv"
        )


def test_empty_env_is_uniform():
    _set_env("")
    assert get_tq_layer_bits_map() == {}
    assert resolve_tq_layer_preset(3, "turboquant_3bit_nc") == "turboquant_3bit_nc"


def test_per_layer_override():
    _set_env(
        '{"0": "turboquant_3bit_nc", "4": "turboquant_3bit_nuqv_out1", '
        '"8": "turboquant_k8v4"}'
    )
    try:
        m = get_tq_layer_bits_map()
        assert m == {
            0: "turboquant_3bit_nc",
            4: "turboquant_3bit_nuqv_out1",
            8: "turboquant_k8v4",
        }
        assert resolve_tq_layer_preset(0, "turboquant_3bit_nuqv") == "turboquant_3bit_nc"
        assert (
            resolve_tq_layer_preset(4, "turboquant_3bit_nuqv")
            == "turboquant_3bit_nuqv_out1"
        )
        # Unlisted layer falls back to the model-level default.
        assert (
            resolve_tq_layer_preset(5, "turboquant_3bit_nuqv")
            == "turboquant_3bit_nuqv"
        )
    finally:
        _set_env(None)


def test_invalid_json_raises():
    _set_env("{not json}")
    try:
        _expect_raises(ValueError, get_tq_layer_bits_map)
    finally:
        _set_env(None)


def test_unknown_preset_raises():
    _set_env('{"0": "turboquant_does_not_exist"}')
    try:
        _expect_raises(ValueError, get_tq_layer_bits_map)
    finally:
        _set_env(None)


def test_non_integer_key_raises():
    _set_env('{"layer0": "turboquant_3bit_nc"}')
    try:
        _expect_raises(ValueError, get_tq_layer_bits_map)
    finally:
        _set_env(None)


def test_non_object_raises():
    _set_env('["turboquant_3bit_nc"]')
    try:
        _expect_raises(ValueError, get_tq_layer_bits_map)
    finally:
        _set_env(None)


def test_same_slot_size_presets_are_single_group_safe():
    # nc and nuqv keep identical slot size -> a mixed map stays in one KV cache
    # group (merge() single-slot-size assert passes).
    nc = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nc", HEAD_DIM)
    nuqv = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    assert nc.slot_size_aligned == nuqv.slot_size_aligned


def test_size_changing_presets_force_separate_groups():
    # out1 grows the slot -> different real page size -> a distinct KV cache
    # group from same-size presets (grouped by frozen-spec equality).
    nuqv = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv", HEAD_DIM)
    out1 = TurboQuantConfig.from_cache_dtype("turboquant_3bit_nuqv_out1", HEAD_DIM)
    assert out1.slot_size_aligned != nuqv.slot_size_aligned


def test_resolver_preserves_slot_size_math():
    # Resolved preset drives the slot size used for allocation.
    _set_env('{"11": "turboquant_k8v4"}')
    try:
        resolved = resolve_tq_layer_preset(11, "turboquant_3bit_nuqv")
        cfg = TurboQuantConfig.from_cache_dtype(resolved, HEAD_DIM)
        ref = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", HEAD_DIM)
        assert cfg.slot_size_aligned == ref.slot_size_aligned
    finally:
        _set_env(None)


if __name__ == "__main__":
    from _run import run_module

    raise SystemExit(1 if run_module(globals()) else 0)
