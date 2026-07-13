# club-dev1060-cherry — Wave-3 custom vLLM build branch

Date: 2026-07-13
Branch: `club-dev1060-cherry`
Last code commit: `80792e9a4f7617ce43ee96eaba9c153fb58ff249` (branch head is this
notes commit on top)
Base: `9e57de7197f234f9d9187715d96e07e007048c0f` (validated pin, main @ 2026-07-13,
"[CPU] Create Proper Numa topology for s390x (#40714)")

Diffstat 9e57de71..HEAD: 17 files changed, 1672 insertions(+), 40 deletions(-).
One commit per PR pick. Not pushed anywhere.

## PRs applied (in pick order)

| # | PR | Commit on branch | PR head | Verdict |
|---|----|------------------|---------|---------|
| 1 | #48361 hybrid-Mamba prefix-cache corruption under MTP/EAGLE (#43559/#47194) | `310495614` | `86e1b161b` (8 commits, squashed) | conflicts-resolved (see below) |
| 2 | #45660 zero-init Marlin GEMM output/reduce buffers (CUDA-graph safety) | `35dd6b785` | `1c55ba5f6` | clean cherry-pick |
| 3 | #48475 clamp spec-decode state index for zero num_accepted_tokens (GDN/FLA) | `a4540dbb2` | `f70b0ffe6` | clean cherry-pick |
| 4 | #46461 TurboQuant continuation guard in attention fast-path | `4b77f3b93` | `d729d7448` (3 commits, squashed) | clean as squashed net diff (per-commit picks conflicted on intermediate states) |
| 5 | #43650 MTP + prefix caching + mamba accuracy fix | — | `3812af4ed` | **SKIPPED — subsumed by #48361** (see below) |
| 6 | #47574 zero new KV blocks for quantized + block-dropping hybrid | `80792e9a4` | `a7f997abe` | clean cherry-pick |

## Conflicts resolved (#48361)

The pin base (07-13) is one day newer than the PR's merge-base (07-12
`1ef1c7ebb`); main gained partial-hash-hit and mamba-partial-tail features in
between. Applied as a squashed 3-way apply of the net PR diff. 9 of 11 files
applied cleanly (incl. the mooncake connector — nothing dropped despite the
single-GPU waiver). Three hunks conflicted:

1. `vllm/v1/core/kv_cache_coordinator.py`, `find_longest_cache_hit_per_group`:
   - Took PR side: `drop_eagle_block=use_eagle and spec.supports_eagle_cache_peek`
     (the semantic gate — recurrent-state specs must not join the eagle
     drop-last-block peek).
   - Kept base side: `alignment_tokens=self._cache_hit_alignment_tokens`
     (base's partial-hash-hit generalization; the PR's
     `self.scheduler_block_size` is the degenerate case of it).
2. `vllm/v1/core/sched/scheduler.py`, `_mamba_block_aligned_split` hunk 1:
   kept base's superset logic (mid-block-resume `next_boundary` branch + chunk
   END alignment — semantically identical to the PR's `round_down` form);
   switched the arithmetic to the PR's `round_down` helper for consistency.
3. `vllm/v1/core/sched/scheduler.py`, `_mamba_block_aligned_split` hunk 2:
   - Kept base's `elif self.mamba_partial_cache_hit:` tail-boundary branch
     (base-only feature, unknown to the PR).
   - Appended the PR's new `elif num_computed_tokens_after_sched < prefill_end:`
     branch after it — the core #43559 fix (a non-final chunk must never end
     mid-block past the last cacheable boundary, else the hashed boundary slot
     is poisoned).
   - Deleted base's trailing "Marconi cache admission" re-round block: the PR
     intentionally moves that cap BEFORE alignment (that relocation hunk
     applied cleanly above) because cap-then-re-round reintroduces unaligned
     non-final chunk ends — exactly the #43559 geometry.

Branch-chain note: with `mamba_partial_cache_hit` active and its inner straddle
condition false, the new `< prefill_end` branch is shadowed for that step. This
is safe: past `last_cache_position` the only hashed state under the
partial-tail feature is the tail entry at `tail_boundary`, and the branch stops
exactly there when it is crossed, so the boundary slot is written aligned.

## Hunks dropped

None. (The mooncake/PD-disagg connector changes from #48361 were pre-authorized
to drop but applied cleanly, so they are in.)

## Why #43650 was skipped

#43650 decrements `max_num_blocks` in `MambaManager.find_longest_cache_hit`
under `use_eagle` so the eagle drop-last-block never removes a mamba state
block. On this branch #48361 fixes the same failure structurally:
- `MambaSpec.supports_eagle_cache_peek` returns `False`
  (`vllm/v1/kv_cache_interface.py`), so `drop_eagle_block` is never `True` for
  mamba groups (both coordinators gate on it).
- The hybrid coordinator gives mamba no eagle lookahead margin
  (`if drop_eagle_block and not isinstance(spec, MambaSpec)`), so the mamba
  finder searches only up to the already-eagle-dropped full-attention hit
  length and its state block at the hit boundary is never popped.
Both mechanisms subsume #43650's +6 lines; applying it on top would only
shorten mamba hits by one extra block for no correctness gain.

## Verification performed

- `python3 -m py_compile` + `ast.parse` on all 15 touched `.py` files: OK.
- No conflict markers anywhere in the tree.
- `marlin.cu` (#45660): eyeballed both hunks — `zero_(c)` inside the existing
  `use_atomic_add` allocation branch, `zero_(c_tmp)` inside the existing
  `use_fp32_reduce` branch; braces balanced, +10/-0, no orphan code.
- Call-signature cross-checks: `find_longest_cache_hit` returns
  `(blocks, hit_length)` and takes `drop_eagle_block`/`alignment_tokens` in all
  managers; `supports_fine_grained_hash_lookup`, `round_down` import, and
  `KVQuantMode`/`ChunkedLocalAttentionSpec` references all resolve.
- NOT verified: CUDA build, runtime tests (no GPU build here by design). The
  PR-carried tests to run on the wheel:
  `tests/v1/e2e/test_hybrid_mamba_prefix_cache_correctness.py`,
  `tests/v1/core/test_mamba_align_chunk_split.py`,
  `tests/kernels/quantization/test_marlin_gemm.py -k zero`,
  `tests/kernels/test_fused_sigmoid_gating_delta_rule.py`.

## Semantic caveat (#46461)

The continuation guard compares per-request `q_len != seq_len` using
`attn_metadata.seq_lens_cpu`, which this metadata builder fills from
`cam.seq_lens_cpu_upper_bound` (`turboquant_attn.py` ~line 278). An upper bound
can exceed the true seq len, flagging a pure first-chunk-prefill batch as
"has continuation" and skipping the flash-attn fast path. That is
correctness-safe (falls back to the slow path) but may cost prefill throughput;
watch for a fast-path hit-rate drop after deploy.

## Recommended wheel-build inputs (GH-Actions / vast.ai)

- **Python 3.12** — `requires-python = ">=3.10,<3.15"` allows 3.13/3.14, but
  3.12 is the docs/CI default and the safest for prebuilt triton/flashinfer
  deps; also matches the historical wheel pattern.
- **CUDA 13.0 (13.0.2)** — `docker/Dockerfile` default is `CUDA_VERSION=13.0.2`
  and CMake's newest arch table activates at nvcc >= 13.0. This is already the
  newest CUDA line main supports; nothing newer to bump to.
- **torch 2.11.0 + cu130** — `pyproject.toml` pins `torch == 2.11.0` exactly
  (and requirements/cuda.txt pins torchvision 0.26.0 / torchaudio 2.11.0
  alongside). "Newest supported" IS the pin; do not bump past it or
  `find_package(Torch)`/ABI checks will fight you.
- **`TORCH_CUDA_ARCH_LIST=8.9`** — per the established wheel pattern.
  ⚠ If this wheel is meant for an RTX **3090** (repo name `club-3090`),
  the correct arch is **8.6**; build `"8.6;8.9"` (+~40% kernel-compile time,
  one wheel serves both) or plain 8.6 if only the 3090 matters. Decide before
  launching the build.
- Env for the build job: `VLLM_USE_PRECOMPILED` **unset** (we changed
  `csrc/…/marlin.cu`, a full source build is mandatory),
  `MAX_JOBS` sized to the runner, `CMAKE_BUILD_TYPE=Release`.

## Reproduce

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm.git
cd vllm && git checkout 9e57de7197f234f9d9187715d96e07e007048c0f -b club-dev1060-cherry
# then cherry-pick / apply the five commits above (branch lives at
# /home/user/engines/vllm-build, not pushed)
```
