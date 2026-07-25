# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal pytest-free test runner for the KVQ CPU suite.

The CPU test venv (``~/shared/needfit/lens-venv``) ships torch+numpy but not
pytest, so each ``test_*`` module runs its ``test_*`` callables through this
harness. Compatible with pytest too (pytest ignores ``run_module``).
"""

import traceback


def run_module(g: dict) -> int:
    names = [k for k in sorted(g) if k.startswith("test_") and callable(g[k])]
    passed = failed = 0
    for name in names:
        try:
            g[name]()
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


def expect_raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")
