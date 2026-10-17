// lowbit_unpack.cuh (cuda lane LOWBIT, 2026-10-07): device unpack of SYMMETRIC int3 / int2 Marlin weights into
// IMMA int8 B fragments (kS8 activations x int3 / int2 weights, m16n8k32). Lossless by construction: every output
// byte is the exact two's-complement int8 of the stored value (int3: -4..3, int2: -2..1); no zero point, no float.
//
// LOGICAL LAYOUT (identical to the served kS8 x kU4 Marlin path, so nothing outside the unpack changes):
//   Per thread per k-iteration the served kernel loads one int4 chunk (4 ints, 32 nibbles) and dequant_data() turns
//   int w = 2j+h into two int8x4 registers (p = 0: low nibbles, p = 1: high nibbles). We name the 8 registers
//   R[o], o = 4j + 2h + p (FragB frag_b[h] of MMA (j, h) is {R[4j+2h], R[4j+2h+1]}). With the gptq_marlin_repack
//   is_a_8bit order, byte i of R[o] holds W[k, n] with
//     k = 32*kt + 4*(lane%4) + 16*(o%2) + i,   n = 32*nt + 16*(o/4) + 8*((o/2)%2) + lane/4.
//   lowbit_pack.py builds R from W with exactly this map and encodes it into the words below.
//
// PACKED WORDS (per thread per k-iteration): int3 = 3 x u32 (12 B), int2 = 2 x u32 (8 B).
// The unpack uses PRMT's sign-replicate mode as a 2-bit decoder: selector nibble [s t x x] -> byte
//   s = 0: LUT[t]  (t = nibble bit 2 picks the a/b source register, bits 1:0 pick a byte; all bytes equal)
//   s = 1: sign(LUT[t]) replicated = 0x00 or 0xFF
//   int2: LUT = {+1, -2}  -> (s,t) = (0,0) +1, (0,1) -2, (1,0) 0, (1,1) -1             (all 4 values, 1 op / reg)
//   int3: LUT = {+3, -4}  -> {+3, -4, 0, -1}, then ONE LOP3 xors bit jx (0 or 1) of each byte with a stored bit l:
//         jx = 0 adds {+2, -3, +1, -2}; jx = 1 adds {+1, -2, +2, -3}: all 8 values either way (2 ops / reg).
//   A PRMT selector view needs [s t] at nibble bits [3:2] of its LOW 16 bits (PRMT ignores the high half); the XOR
//   view needs l at byte-lane bit jx. The bit plan below fills every bit of every word (no padding):
//
//   int3  W0 = P word (4 PRMT views, no XOR bits)      W1 = Q word, W2 = R word (2 PRMT views + 4 XOR quads each)
//     R[0] = prmt(W0      ) ^ (W1      & 0x01010101)    R[4] = prmt(W0 << 2 ) ^ (W2      & 0x01010101)
//     R[1] = prmt(W0 >> 16) ^ (W1      & 0x02020202)    R[5] = prmt(W0 >> 14) ^ (W2      & 0x02020202)
//     R[2] = prmt(W1      ) ^ (W1 >> 4 & 0x01010101)    R[6] = prmt(W2      ) ^ (W2 >> 4 & 0x01010101)
//     R[3] = prmt(W1 >> 16) ^ (W1 >> 4 & 0x02020202)    R[7] = prmt(W2 >> 16) ^ (W2 >> 4 & 0x02020202)
//     per 32 weights: 7 SHF + 8 PRMT + 8 LOP3 = 23 ALU (served kU4 + zp: see SASS-COUNTS.txt), 2 constant regs.
//     j = 0 (R[0..3]) reads W0, W1 only; j = 1 (R[4..7]) reads W0, W2 only.
//   int2  W0 -> R[0..3], W1 -> R[4..7]:  R[4q+0] = prmt(W), R[4q+1] = prmt(W >> 16), R[4q+2] = prmt(W << 2),
//         R[4q+3] = prmt(W >> 14).  Per 32 weights: 6 SHF + 8 PRMT = 14 ALU, 2 constant regs.
//
// SHARED / GLOBAL SEGMENT LAYOUT (lowbit_pack.py: to_segments): Marlin's kS8 B rows are k32 x N, N int4 chunks per
// row in units of 32-chunk tiles (one n32 x k32 repack tile = one warp-k-iteration). A lowbit tile keeps the tile
// order and stores, per 32 lanes, plane A = 32 x int2 then (int3 only) plane B = 32 x u32:
//   int3: 384 B = 24 int4 per tile; lane l reads int2 A[l] = {W0, W1} and u32 B[l] = W2 (LDS.64 + LDS.32).
//   int2: 256 B = 16 int4 per tile; lane l reads int2 A[l] = {W0, W1}            (LDS.64).
#pragma once
#include <cstdint>

namespace lowbit {

// PRMT default mode: d.byte[n] = msb-replicate?(src.byte[sel.nib[n] & 7]) with replicate when sel.nib[n] & 8.
// (Inline PTX, not __byte_perm: the intrinsic's documented selector is 3 bits; the sign mode is the point here.)
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

constexpr uint32_t kI3LutA = 0x03030303u;  // t = 0 -> +3
constexpr uint32_t kI3LutB = 0xFCFCFCFCu;  // t = 1 -> -4
constexpr uint32_t kI2LutA = 0x01010101u;  // t = 0 -> +1
constexpr uint32_t kI2LutB = 0xFEFEFEFEu;  // t = 1 -> -2

// int3: the four registers of MMA column block j (R[4j .. 4j+3]) from W0 and W1 (j = 0) or W2 (j = 1).
template <int J>
__device__ __forceinline__ void unpack3_j(uint32_t w0, uint32_t wx, uint32_t* r) {
  const uint32_t a = kI3LutA, b = kI3LutB;
  const uint32_t x4 = wx >> 4;
  if constexpr (J == 0) {
    r[0] = prmt(a, b, w0) ^ (wx & 0x01010101u);
    r[1] = prmt(a, b, w0 >> 16) ^ (wx & 0x02020202u);
  } else {
    r[0] = prmt(a, b, w0 << 2) ^ (wx & 0x01010101u);
    r[1] = prmt(a, b, w0 >> 14) ^ (wx & 0x02020202u);
  }
  r[2] = prmt(a, b, wx) ^ (x4 & 0x01010101u);
  r[3] = prmt(a, b, wx >> 16) ^ (x4 & 0x02020202u);
}

__device__ __forceinline__ void unpack3(const uint32_t* w, uint32_t* r) {
  unpack3_j<0>(w[0], w[1], r);
  unpack3_j<1>(w[0], w[2], r + 4);
}

template <int J>
__device__ __forceinline__ void unpack2_j(uint32_t w, uint32_t* r) {
  const uint32_t a = kI2LutA, b = kI2LutB;
  r[0] = prmt(a, b, w);
  r[1] = prmt(a, b, w >> 16);
  r[2] = prmt(a, b, w << 2);
  r[3] = prmt(a, b, w >> 14);
}

__device__ __forceinline__ void unpack2(const uint32_t* w, uint32_t* r) {
  unpack2_j<0>(w[0], r);
  unpack2_j<1>(w[1], r + 4);
}

// Reference unpacks of the 4-bit formats (verbatim math of dequant.h dequant<int32_t, kU4B8> and
// sub_zp_and_dequant<int32_t, kU4>) so SASS-COUNTS.txt counts all widths with the same harness.
__device__ __forceinline__ void unpack4_sym(const uint32_t* q4, uint32_t* r) {
#pragma unroll
  for (int w = 0; w < 4; w++) {
    uint32_t q = q4[w];
    r[2 * w] = ((q & 0x0F0F0F0Fu | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
    q >>= 4;
    r[2 * w + 1] = ((q & 0x0F0F0F0Fu | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
  }
}
__device__ __forceinline__ void unpack4_zp(const uint32_t* q4, uint32_t qzp, int warp_par, uint32_t* r) {
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int off = warp_par * 2 + j;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint32_t q = q4[2 * j + h];
      const uint32_t zp = (qzp >> (off * 8 + 4 * h)) & 0xF;
      const uint32_t rz = 0x01010101u * zp;
      r[4 * j + 2 * h] = ((q & 0x0F0F0F0Fu | 0x80808080u) - rz) ^ 0x80808080u;
      q >>= 4;
      r[4 * j + 2 * h + 1] = ((q & 0x0F0F0F0Fu | 0x80808080u) - rz) ^ 0x80808080u;
    }
  }
}

}  // namespace lowbit
