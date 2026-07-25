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

TurboQuantConfig = config.TurboQuantConfig
TQ_PRESETS = config.TQ_PRESETS
resolve_tq_layer_preset = config.resolve_tq_layer_preset
get_tq_layer_bits_map = config.get_tq_layer_bits_map
TQ_LAYER_BITS_ENV = config.TQ_LAYER_BITS_ENV
get_centroids = centroids.get_centroids
get_value_codebook = centroids.get_value_codebook
solve_lloyd_max = centroids.solve_lloyd_max
