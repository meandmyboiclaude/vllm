# club-dev1474-cherry — wheel build branch (2026-07-25)

> **Historical record.** The sections below describe the 2026-07-25
> club-dev1474-cherry branch at base 0ba2aa35 and are kept as-is for that
> branch's audit trail. The current branch is club-tip-0b19ebcac-20260822
> (upstream main @ 0b19ebcac + carried picks, vendored open-PR picks, and
> review-fix commits); its authoritative pick-set is the branch commit log
> (`git log 0b19ebcac..HEAD`), not the tables below.

Base: 0ba2aa35a81dcc3246b26291368b53fa2389c7d7 (nightly-0ba2aa35, main 2026-07-25,
0.23.1rc1.dev1474). Successor of club-dev1060-cherry (see its BUILD-NOTES).

## Old-wheel parity audit (the dev1060cherry 7 picks)
| pick | status here |
|---|---|
| #48361 hybrid-Mamba prefix-cache corruption | CORE ABSORBED by main in-range (#48425/#47782/#48481 rework; PR slimmed by author 07-23 to a Mooncake EAGLE cache-peek gate) — residual PICKED (25e7a397b4c, current head 68ada38bc4) |
| #45660 Marlin GEMM zero-init (compiled) | PICKED (558e22c4894, head 1c55ba5f65) — the wheel-only item |
| #48475 GDN spec state-index clamp | PICKED (76c86291e7b, fresh head already on third_party layout) |
| #46461 TQ prefill continuation guard | PICKED (60085ab6bcb) — genesis PN401/PN86 self-retire on its marker |
| #47574 zero KV blocks quantized hybrid | upstream merged 8ce53a616e |
| #48483 cudagraph capture memory | upstream merged 1be6e937b2 |
| #48363 Mamba2 Triton warmup | PICKED (b517ca46e5b, head c65ba6aa8f; kernel_warmup.py auto-merged over #47451 infra) |

All picks auto-merged; py_compile spot-checks clean. CI workflows stripped
(6c3688a0ad1). Perf motivation: the −13% prod-prompt TPS regression on stock
nightly (VLLM-UPDATE-20260725.md) — picks theory tested by the post-build
prod-30 screen.

## Second wave (2026-07-25, scout-agent verdict)
| pick | why |
|---|---|
| #48177 | TQ KV-dtype preserve in reshape (fresh head incl. main merge) |
| #43747 | TQ cudagraph capture crash w/ spec-decode + chunked-prefill (BUG-127-adjacent suspect) — auto-merged over our #46461 pick |
| #48188 | ~6x Mamba chunk-metadata computation (prod long-prefill perf) |
| #49798 | hand-applied essence (import + guard) — PR head was stacked on unrelated multimodal work |
Rejected for today (scout): #46067/#48815/#49738 (conflict stacking risk on scheduler/turboquant), #43642 (competes with #48363), base-bump past 0ba2aa35 (8 irrelevant commits).
