# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated module loaders for TurboQuant KVQ CPU tests.

The full ``vllm`` package requires compiled CUDA extensions
(``vllm._C_stable_libtorch``) that are unavailable in the CPU test venv, so
importing ``vllm.model_executor...`` fails at package import time. The
TurboQuant ``config`` and ``centroids`` modules are themselves pure
stdlib/torch, so we load them directly by file path under synthetic module
names, bypassing ``vllm/__init__``.
"""

import importlib.util
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_TQ = _REPO / "vllm" / "model_executor" / "layers" / "quantization" / "turboquant"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass processing resolves string annotations via
    # sys.modules[cls.__module__].
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


centroids = _load("_tq_centroids", _TQ / "centroids.py")
config = _load("_tq_config", _TQ / "config.py")
sink = _load("_tq_sink", _TQ / "sink.py")

# Kernel sources are read as text (they import triton, unavailable on CPU) so
# tests can assert the device code and the host reference share one definition.
KERNEL_DECODE_SRC = _REPO / "vllm" / "v1" / "attention" / "ops" / (
    "triton_turboquant_decode.py"
)
KERNEL_STORE_SRC = _REPO / "vllm" / "v1" / "attention" / "ops" / (
    "triton_turboquant_store.py"
)
BACKEND_SRC = _REPO / "vllm" / "v1" / "attention" / "backends" / "turboquant_attn.py"

TurboQuantConfig = config.TurboQuantConfig
TQ_PRESETS = config.TQ_PRESETS
resolve_tq_layer_preset = config.resolve_tq_layer_preset
get_tq_layer_bits_map = config.get_tq_layer_bits_map
TQ_LAYER_BITS_ENV = config.TQ_LAYER_BITS_ENV
get_centroids = centroids.get_centroids
get_value_codebook = centroids.get_value_codebook
solve_lloyd_max = centroids.solve_lloyd_max
SINK_HASH_MULT = sink.SINK_HASH_MULT
SINK_HASH_SHIFT = sink.SINK_HASH_SHIFT
SINK_EMPTY_TAG = sink.SINK_EMPTY_TAG
SINK_OVERPROVISION_ENV = sink.SINK_OVERPROVISION_ENV
sink_cache_slots = sink.sink_cache_slots
sink_row_for_slot = sink.sink_row_for_slot
sink_eligible = sink.sink_eligible
sink_lookup = sink.sink_lookup
resolve_tag_claims = sink.resolve_tag_claims
build_sink_spec = sink.build_sink_spec
SinkBufferSpec = sink.SinkBufferSpec
