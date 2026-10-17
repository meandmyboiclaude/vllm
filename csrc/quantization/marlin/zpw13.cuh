// [ZPW13 cuda-zpw13 2026-10-15] 13-bit ZPP group words (DESIGN-ZPW13.v1 §5, v1.3 decode §9): the integer helpers of the ZPW13 twins
// (marlin_template.h, MARLIN_ZPW13_TWIN) and of the CPU test (wip/zpw13/hosttest), which compiles this same text for the host.
// Included by marlin_template.h only in a ZPW13 TU, so no other TU's token stream changes. ZPW13_HD defaults to __device__ __forceinline__.
#pragma once
#ifndef ZPW13_HD
  #define ZPW13_HD __device__ __forceinline__
#endif
namespace zpw13 {
// 12-bit code field e (0..7) of a lane's 96-bit chunk {w0, w1, w2} = its code words 0, 1, 2 (field e = bits 12 e .. 12 e + 11, i.e.
// halfword e of the served ZPP chunk); fields 2 and 5 straddle a word. e is a constant after the k-loop unroll.
ZPW13_HD uint32_t field(int e, uint32_t w0, uint32_t w1, uint32_t w2) {
  switch (e) {
    case 0: return w0 & 0xFFFu;
    case 1: return (w0 >> 12) & 0xFFFu;
    case 2: return ((w0 >> 24) | (w1 << 8)) & 0xFFFu;
    case 3: return (w1 >> 4) & 0xFFFu;
    case 4: return (w1 >> 16) & 0xFFFu;
    case 5: return ((w1 >> 28) | (w2 << 4)) & 0xFFFu;
    case 6: return (w2 >> 8) & 0xFFFu;
    default: return w2 >> 20;
  }
}
// b = a lane group's 4 zero points as bytes (byte i = z0 + the flip of word i, <= 15): 0x01010101 * byte i, one PRMT replicating it.
ZPW13_HD uint32_t rzb(uint32_t b, int i) { return __byte_perm(b, 0u, 0x1111u * static_cast<uint32_t>(i)); }
// dequant.h sub_zp_and_dequant<int32_t, kU4, true> (INT4 with zp -> INT8) with its subtrahend zp * 0x01010101 passed in (rz = rzb(b, i)).
// Unsigned arithmetic: the same bits as the served signed int body (wrapping subtraction; q >> 4 differs only in bits 28..31, which
// the mask clears).
ZPW13_HD void sub_rz_and_dequant(int q, int32_t* frag_b, uint32_t rz) {
  const uint32_t MASK = 0x80808080u;
  const uint32_t uq = static_cast<uint32_t>(q);
  frag_b[0] = static_cast<int32_t>((((uq & 0x0F0F0F0Fu) | MASK) - rz) ^ MASK);
  frag_b[1] = static_cast<int32_t>(((((uq >> 4) & 0x0F0F0F0Fu) | MASK) - rz) ^ MASK);
}
}  // namespace zpw13
