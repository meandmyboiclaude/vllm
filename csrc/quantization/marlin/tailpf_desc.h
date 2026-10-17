// [0043 TAILPF] The next-launch descriptor of the tail-prefetch twins, shared by marlin.cu (host, under #if MARLIN_TAILPF)
// and marlin_template.h (device, only in the TUs that define MARLIN_TPF_TWIN 1). Both include it INSIDE namespace
// MARLIN_NAMESPACE_NAME, so it holds declarations only (no #include). Pure integer code: tailpf_host_check.cpp compiles it
// on the host and checks marlin_tpf_first_tile + marlin_tpf_b_line against a literal port of the kernel's init_slice() and
// fetch_to_shared(), and marlin_tpf_q against the division it replaces.
#pragma once

#ifndef MARLIN_TPF_HD
  #if defined(__CUDACC__)
    #define MARLIN_TPF_HD __host__ __device__
  #else
    #define MARLIN_TPF_HD
  #endif
#endif

// n / d for 0 <= n < 2^31 and 1 <= d < 2^31 without a division instruction: q = (n * mul) >> shift with shift = 31 +
// ceil(log2 d) and mul = ceil(2^shift / d) < 2^32 (d == 1: mul = 2^31, shift = 31, exact). Error term n * (mul * d - 2^shift)
// / (d * 2^shift) < n / 2^shift < 2^-ceil(log2 d) <= 1 / d, so the floor never moves (Granlund-Montgomery; cutlass
// FastDivmod). On the device: one IMAD.WIDE.U32 + one 64-bit shift (no I2F / MUFU.RCP / F2I reciprocal sequence).
struct MarlinTpfDiv {
  int d;               // the divisor (0 = unused: q = 0)
  unsigned int mul;
  unsigned int shift;
};

MARLIN_TPF_HD inline MarlinTpfDiv marlin_tpf_div(int d) {
  MarlinTpfDiv v{d, 0u, 32u};
  if (d <= 0) return v;
  unsigned int l = 0;
  while ((1ULL << l) < static_cast<unsigned long long>(d)) l++;  // ceil(log2 d)
  v.shift = 31u + l;
  v.mul = static_cast<unsigned int>(((1ULL << v.shift) + static_cast<unsigned long long>(d) - 1) /
                                    static_cast<unsigned long long>(d));
  return v;
}

MARLIN_TPF_HD inline int marlin_tpf_q(int n, const MarlinTpfDiv& v) {
  return static_cast<int>((static_cast<unsigned long long>(static_cast<unsigned int>(n)) * v.mul) >> v.shift);
}

// What a TPF twin prefetches at its end: the NEXT Marlin launch's group scales and grouped zero points (flat 128-B line
// ranges, grid-strided over this launch's threads) and, with b != nullptr, the first `stages` k-tiles of B that each CTA of
// the next launch fetches first. Passed BY VALUE as the twin's first kernel parameter, so a CUDA graph bakes it into the
// kernel node (marlin.cu may re-set the node's parameters while the capture is still in progress). All-null = no prefetch.
// Contract (marlin.cu kt_tpf_self_desc, host proof D): s_lines == 0 when s == nullptr, zp_lines == 0 when zp == nullptr,
// b == nullptr and grid == 0 without a B part; with a B part iters * grid < 2^31 and p2_flat < 2^31 (32-bit index math).
struct MarlinTpfDesc {
  const int4* s;    // next launch's group scales, rounded down to a 128-B line (nullptr: none)
  const int4* zp;   // next launch's grouped zero points, rounded down to a 128-B line (nullptr: none)
  const int4* b;    // next launch's B (nullptr: no first-stage prefetch)
  int s_lines;      // 128-B lines from s
  int zp_lines;     // 128-B lines from zp
  int evict_last;   // 1 = prefetch.global.L2::evict_last, 0 = ::evict_normal
  // the next launch's partition (marlin_template.h's expressions), valid when b != nullptr
  int grid;         // its gridDim.x
  int k_tiles;      // prob_k / 16 / thread_k_blocks
  int n_tiles;      // prob_n / 16 / thread_n_blocks
  int part1_tiles;  // global_mn_tiles - part2_mn_tiles (full-k tiles before the part-2 stripe; a multiple of grid)
  int p2_flat;      // k_tiles * part2_mn_tiles
  int iters;        // k-tiles per part-2 stripe (with the group rounding)
  int p1_q, p1_r;   // the [0033] tile-major part-1 map
  int tile_major;   // bit 0 of its tile_major argument
  int row_int4;     // int4s per B stage row (b_sh_stride), a multiple of 8
  int rows;         // B rows per k-tile (thread_k_blocks / (a8 ? 2 : 1))
  int gl_stride;    // int4s between B rows (b_gl_stride), a multiple of 8
  int stages;       // k-tiles to prefetch from each CTA's first tile (1..3)
  // the divisors of the device index math (marlin_tpf_finish fills them from the fields above)
  MarlinTpfDiv div_n_tiles;  // n_tiles
  MarlinTpfDiv div_p1_q1;    // p1_q + 1
  MarlinTpfDiv div_p1_q;     // p1_q (0 = unused: then no part-1 CTA reaches the third tile-major branch)
  MarlinTpfDiv div_k_tiles;  // k_tiles
  MarlinTpfDiv div_lpr;      // row_int4 / 8 = 128-B lines per B row
};

struct MarlinTpfTile {
  int col;  // column tile
  int row;  // first k-tile
  int n;    // k-tiles to prefetch (0: the CTA has no tile)
};

// The first tile CTA yb of the launch `d` describes works on, i.e. what the kernel's first init_slice() sets for
// blockIdx.x == yb: a part-1 full-k tile (slice_col_par = yb, through the [0033] map) when the launch has part-1 tiles, else
// its part-2 stripe start (slice_col_par = iters * yb / k_tiles, slice_row = iters * yb % k_tiles), clipped like
// init_part2_slice (no tile past part2_mn_tiles; slice_iters = iters cut at the column end). n = min(stages, slice_iters).
// Every quotient is marlin_tpf_q (all dividends are >= 0 and < 2^31 by the descriptor contract).
MARLIN_TPF_HD inline MarlinTpfTile marlin_tpf_first_tile(const MarlinTpfDesc& d, int yb) {
  MarlinTpfTile t{0, 0, 0};
  int iters_here;
  if (yb < d.part1_tiles) {
    if (!d.tile_major)
      t.col = yb - marlin_tpf_q(yb, d.div_n_tiles) * d.n_tiles;
    else if (yb < d.p1_r * (d.p1_q + 1))
      t.col = marlin_tpf_q(yb, d.div_p1_q1);
    else
      t.col = d.p1_r + marlin_tpf_q(yb - d.p1_r * (d.p1_q + 1), d.div_p1_q);
    t.row = 0;
    iters_here = d.k_tiles;
  } else {
    const int f = d.iters * yb;
    if (f >= d.p2_flat) return t;
    const int c = marlin_tpf_q(f, d.div_k_tiles);
    t.row = f - c * d.k_tiles;
    const int cp = c + d.part1_tiles;
    t.col = cp - marlin_tpf_q(cp, d.div_n_tiles) * d.n_tiles;
    iters_here = d.iters;
    if (t.row + iters_here > d.k_tiles) iters_here = d.k_tiles - t.row;
  }
  t.n = iters_here < d.stages ? iters_here : d.stages;
  return t;
}

// The int4 offset from d.b of 128-B line i (0 <= i < t.n * d.rows * d.div_lpr.d) of tile t's first t.n k-tiles: row
// t.row * d.rows + i / lpr of the stage rows (stride gl_stride), 8 int4s per line, column tile t.col (row_int4 int4s each).
MARLIN_TPF_HD inline long long marlin_tpf_b_line(const MarlinTpfDesc& d, const MarlinTpfTile& t, int i) {
  const int r = marlin_tpf_q(i, d.div_lpr);
  return static_cast<long long>(d.row_int4) * t.col + static_cast<long long>(d.gl_stride) * (t.row * d.rows + r) +
         8 * (i - r * d.div_lpr.d);
}

// Fill the partition fields of `d` for a launch of grid CTAs, from the kernel's own expressions (marlin_template.h: parallel,
// k_tiles, n_tiles, part2_mn_tiles, part1 map, iters with the group rounding). fold_parallel: a MarlinFOLD launch
// (ceil(prob_m / m_block_size) problem copies) instead of the served floor. Host only.
MARLIN_TPF_HD inline void marlin_tpf_partition(MarlinTpfDesc& d, int prob_m, int prob_n, int prob_k, int m_block_size,
                                               bool fold_parallel, int thread_k_blocks, int thread_n_blocks,
                                               int group_blocks, int grid, int tile_major_bit) {
  int parallel = 1;
  if (prob_m > m_block_size)
    parallel = fold_parallel ? (prob_m + m_block_size - 1) / m_block_size : prob_m / m_block_size;
  d.k_tiles = prob_k / 16 / thread_k_blocks;
  d.n_tiles = prob_n / 16 / thread_n_blocks;
  const int global_mn_tiles = parallel * d.n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  if (global_mn_tiles > grid) {
    part2_mn_tiles = global_mn_tiles % grid;
    if (part2_mn_tiles * 3 <= grid) part2_mn_tiles += grid;
  }
  d.part1_tiles = global_mn_tiles - part2_mn_tiles;
  d.p1_q = d.part1_tiles / d.n_tiles;
  d.p1_r = d.part1_tiles % d.n_tiles;
  d.p2_flat = d.k_tiles * part2_mn_tiles;
  int iters = (d.k_tiles * part2_mn_tiles + grid - 1) / grid;
  if (group_blocks != -1 && group_blocks >= thread_k_blocks) {
    const int r = group_blocks / thread_k_blocks;
    iters = r * ((iters + r - 1) / r);
  }
  d.iters = iters;
  d.grid = grid;
  d.tile_major = tile_major_bit;
}

// The divisors of the device index math, from the partition fields (host only; after marlin_tpf_partition and row_int4).
MARLIN_TPF_HD inline void marlin_tpf_finish(MarlinTpfDesc& d) {
  d.div_n_tiles = marlin_tpf_div(d.n_tiles);
  d.div_p1_q1 = marlin_tpf_div(d.p1_q + 1);
  d.div_p1_q = marlin_tpf_div(d.p1_q);
  d.div_k_tiles = marlin_tpf_div(d.k_tiles);
  d.div_lpr = marlin_tpf_div(d.row_int4 / 8);
}
