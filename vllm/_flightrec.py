# SPDX-License-Identifier: Apache-2.0
"""Genesis flight recorder — crash-durable low-level event log.

House diagnostic module (rebase-6, 2026-08-22). Design goals, in order:

1. CRASH-DURABLE: every event lands in a HOST-mounted file (default
   /profiles/flightrec/) the moment it happens (line-buffered JSONL, fsync
   on error-class events). The engine, the container, or podman itself can
   die at any point and everything already written survives on the host —
   nothing depends on the container staying up.
2. LOW-LEVEL PATH TRACING: `ev()` one-shot events, `path()` enter/exit
   spans, `scan()` tensor summaries (shape/dtype/device/min/max/nan count)
   for NaN and corruption hunts.
3. CHEAP WHEN QUIET: FLIGHTREC=1 keeps a bounded in-memory ring and writes
   only WARN+ events; the full ring is flushed to the host file on ANY
   crash (excepthook, threading hook, atexit-with-error). FLIGHTREC=2
   streams everything as it happens.

Env knobs (all read once at first use):
  FLIGHTREC        0=off (default) | 1=flight ring | 2=full stream
  FLIGHTREC_DIR    sink dir, default /profiles/flightrec (must be a host
                   bind mount for crash durability)
  FLIGHTREC_RING   ring size at level 1, default 4096

Native crashes (SIGSEGV / CUDA abort) are covered by faulthandler wired to
a dedicated always-open file in the same dir: python-frame dumps for every
thread survive even when the interpreter cannot run another line.

Capture safety: scan() refuses to touch tensor VALUES inside dynamo
compilation or CUDA-graph capture (shape/dtype/device only) — an unguarded
sync or print on a captured path is itself a crash class (PN409 lesson).
"""
from __future__ import annotations

import atexit
import collections
import faulthandler
import json
import os
import sys
import threading
import time

_LOCK = threading.Lock()
_STATE: dict = {"level": None, "fh": None, "ring": None, "n": 0, "pid": None}


def _level() -> int:
    lv = _STATE["level"]
    if lv is None:
        try:
            lv = int(os.environ.get("FLIGHTREC", "0").strip() or "0")
        except ValueError:
            lv = 0
        _STATE["level"] = lv
        if lv > 0:
            _open_sink()
    return lv


def _open_sink() -> None:
    d = os.environ.get("FLIGHTREC_DIR", "/profiles/flightrec")
    try:
        os.makedirs(d, exist_ok=True)
        pid = os.getpid()
        _STATE["pid"] = pid
        path = os.path.join(d, f"flightrec-{pid}-{int(time.time())}.jsonl")
        # buffering=1 => every line hits the OS immediately; the host file
        # then survives any in-container death.
        _STATE["fh"] = open(path, "a", buffering=1, encoding="utf-8")
        try:
            ring_n = int(os.environ.get("FLIGHTREC_RING", "4096") or "4096")
        except ValueError:
            ring_n = 4096
        _STATE["ring"] = collections.deque(maxlen=max(ring_n, 64))
        # Native-crash coverage: dedicated fd, all threads.
        fh_path = os.path.join(d, f"faulthandler-{pid}.log")
        _STATE["fh_native"] = open(fh_path, "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_STATE["fh_native"], all_threads=True)
        _install_hooks()
        _write_now({"t": time.time(), "tag": "flightrec.start", "pid": pid,
                    "argv": sys.argv[:8]}, fsync=True)
    except Exception as e:  # noqa: BLE001 — the recorder must never kill the host
        sys.stderr.write(f"[flightrec] sink unavailable: {e}\n")
        _STATE["level"] = 0


def _write_now(row: dict, fsync: bool = False) -> None:
    fh = _STATE.get("fh")
    if fh is None:
        return
    try:
        fh.write(json.dumps(row, default=str) + "\n")
        _STATE["n"] += 1
        if fsync or _STATE["n"] % 200 == 0:
            os.fsync(fh.fileno())
    except Exception:  # noqa: BLE001
        pass


def _maybe_arm_memhist() -> None:
    """Arm torch CUDA allocator history (FLIGHTREC_MEMHIST=1, diag boots only).

    Every later allocation records its python stack; crash() then dumps the
    full snapshot so an OOM attributes byte-for-byte to the allocating call.
    """
    if _STATE.get("memhist") or os.environ.get("FLIGHTREC_MEMHIST", "0") != "1":
        return
    try:
        import torch
        torch.cuda.memory._record_memory_history(max_entries=200000)
        _STATE["memhist"] = True
        ev("memhist.armed")
    except Exception as e:  # noqa: BLE001
        ev("memhist.arm_failed", level="warn", exc=repr(e))
        _STATE["memhist"] = False


def ev(tag: str, level: str = "info", **fields) -> None:
    """One event. level: info|warn|error. error always streams + fsyncs."""
    lv = _level()
    if lv == 0:
        return
    if tag == "runner.load_model.enter":
        _maybe_arm_memhist()
    row = {"t": time.time(), "tag": tag, "lvl": level, **fields}
    with _LOCK:
        if lv >= 2 or level in ("warn", "error"):
            _write_now(row, fsync=(level == "error"))
        elif _STATE.get("ring") is not None:
            _STATE["ring"].append(row)


class path:  # noqa: N801 — tiny context manager, lowercase by design
    """with flightrec.path("kv.alloc", groups=3): ... — enter/exit span."""

    def __init__(self, tag: str, **fields):
        self.tag, self.fields = tag, fields

    def __enter__(self):
        self.t0 = time.time()
        ev(self.tag + ".enter", **self.fields)
        return self

    def __exit__(self, et, e, tb):
        if et is not None:
            ev(self.tag + ".raise", level="error", exc=repr(e),
               dur_ms=round((time.time() - self.t0) * 1e3, 3), **self.fields)
        else:
            ev(self.tag + ".exit",
               dur_ms=round((time.time() - self.t0) * 1e3, 3), **self.fields)
        return False


def scan(t, tag: str, values: bool = True) -> None:
    """Tensor summary event. Values (min/max/nan) only when SAFE and asked."""
    if _level() == 0:
        return
    try:
        row = {"shape": tuple(t.shape), "dtype": str(t.dtype),
               "dev": str(t.device)}
        if values:
            import torch
            capturing = False
            try:
                capturing = (torch.compiler.is_compiling()
                             or torch.cuda.is_current_stream_capturing())
            except Exception:  # noqa: BLE001
                capturing = True
            if not capturing and t.is_floating_point():
                f = t.detach().float()
                row["nan"] = int(f.isnan().sum())
                row["inf"] = int(f.isinf().sum())
                fin = f[f.isfinite()]
                if fin.numel():
                    row["min"] = float(fin.min())
                    row["max"] = float(fin.max())
        ev(tag, level=("warn" if row.get("nan") or row.get("inf") else "info"),
           **row)
    except Exception as e:  # noqa: BLE001
        ev(tag + ".scan_failed", level="warn", exc=repr(e))


def crash(exc: BaseException, where: str = "") -> None:
    """Dump the ring + the exception durably. Call from any except path."""
    if _level() == 0:
        return
    import traceback
    with _LOCK:
        ring = _STATE.get("ring") or ()
        for row in ring:
            _write_now(row)
        if ring:
            try:
                ring.clear()
            except Exception:  # noqa: BLE001
                pass
        _write_now({"t": time.time(), "tag": "crash", "lvl": "error",
                    "where": where, "exc": repr(exc),
                    "tb": traceback.format_exc(limit=40)}, fsync=True)
        if _STATE.get("memhist"):
            # Full allocator snapshot (every live alloc + python stack) —
            # the whatOOMed() raw input. Host-durable like everything else.
            try:
                import torch
                d = os.environ.get("FLIGHTREC_DIR", "/profiles/flightrec")
                snap = os.path.join(d, f"memsnap-{_STATE.get('pid')}.pickle")
                torch.cuda.memory._dump_snapshot(snap)
                _write_now({"t": time.time(), "tag": "memhist.dumped",
                            "lvl": "error", "path": snap}, fsync=True)
            except Exception as e:  # noqa: BLE001
                _write_now({"t": time.time(), "tag": "memhist.dump_failed",
                            "lvl": "error", "exc": repr(e)}, fsync=True)


def _install_hooks() -> None:
    prev_hook = sys.excepthook

    def _hook(et, e, tb):
        crash(e, where="sys.excepthook")
        prev_hook(et, e, tb)

    sys.excepthook = _hook
    prev_thook = threading.excepthook

    def _thook(args):
        crash(args.exc_value, where=f"thread:{args.thread.name}")
        prev_thook(args)

    threading.excepthook = _thook

    @atexit.register
    def _final():  # noqa: ANN202
        with _LOCK:
            for row in _STATE.get("ring") or ():
                _write_now(row)
            _write_now({"t": time.time(), "tag": "flightrec.exit"}, fsync=True)
