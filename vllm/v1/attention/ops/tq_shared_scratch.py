# SPDX-License-Identifier: Apache-2.0
"""Shared TurboQuant decode scratch (BUG-199).

One high-water scratch set per device for the eager/fallback decode path.
Layers execute sequentially, so a single set suffices; per-caller retention
(buf_holder=layer) kept n_layers x 3 fp32 buffers alive at high-water and
never reused them. Callers that pass explicit buffers (cudagraph fixed
buffers, workspace manager views) are untouched — this cache only serves
calls that would otherwise allocate fresh.
"""
from types import SimpleNamespace

_BY_DEVICE: dict[int, SimpleNamespace] = {}


def shared_scratch(device_index: int) -> SimpleNamespace:
    ns = _BY_DEVICE.get(device_index)
    if ns is None:
        ns = _BY_DEVICE[device_index] = SimpleNamespace()
    return ns


def reset_shared_scratch() -> None:
    _BY_DEVICE.clear()
