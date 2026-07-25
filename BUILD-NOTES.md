# club-dev1474-cherry — wheel build branch (2026-07-25)

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
