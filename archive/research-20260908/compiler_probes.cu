// Compiler-only evidence: no host main, no launches, no binary mutations.
#include <stdint.h>
#include <cuda_fp16.h>

#define PROBE(name, ...) \
extern "C" __global__ void name(const uint32_t *in, uint32_t *out) { \
    const uint32_t a = in[0], b = in[1], c = in[2]; \
    uint32_t r; \
    __VA_ARGS__ \
    out[0] = r; \
    out[1] = a; out[2] = b; out[3] = c; \
}

PROBE(dot4_cpp,
    r = c;
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) {
        int av = static_cast<int8_t>(a >> (8*j));
        int bv = static_cast<int8_t>(b >> (8*j));
        r += av * bv;
    }
)

PROBE(dot4_vmad,
    r = c;
    asm volatile(
        "vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
)

PROBE(vmad_u16,
    asm volatile("vmad.u32.u32.u32 %0, %1.h0, %2.h1, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(vadd_byte_merge,
    asm volatile("vadd.u32.u32.u32 %0.b1, %1.b0, %2.b2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(vadd4_packed,
    asm volatile("vadd4.u32.u32.u32 %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(sad4_u8,
    asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(sad4_s8,
    asm volatile("vabsdiff4.s32.s32.s32.add %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(absdiff4_u8,
    asm volatile("vabsdiff4.u32.u32.u32 %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(prmt_byte_signs,
    asm volatile("prmt.b32 %0, %1, %2, 0xba98;"
                 : "=r"(r) : "r"(a), "r"(b));
)

PROBE(i8_to_f32,
    float f = static_cast<float>(static_cast<int8_t>(a >> 16));
    r = __float_as_uint(f);
)

PROBE(i8_to_f32_mov_ptx,
    float f;
    asm volatile("{ .reg .b16 lo, hi; mov.b32 {lo,hi}, %1; "
                 "cvt.rn.f32.s8 %0, hi; }" : "=f"(f) : "r"(a));
    r = __float_as_uint(f);
)

PROBE(i8_to_f32_bfe_ptx,
    float f;
    asm volatile("{ .reg .s32 t; bfe.s32 t, %1, 16, 8; "
                 "cvt.rn.f32.s32 %0, t; }" : "=f"(f) : "r"(a));
    r = __float_as_uint(f);
)

PROBE(i8_to_f16,
    __half h = __int2half_rn(static_cast<int8_t>(a >> 16));
    r = __half_as_ushort(h);
)

PROBE(f16_to_f32,
    r = __float_as_uint(__half2float(__ushort_as_half(a & 0xffff)));
)

PROBE(f32_pair_to_f16x2,
    __half2 h = __floats2half2_rn(__uint_as_float(a), __uint_as_float(b));
    r = reinterpret_cast<const uint32_t &>(h);
)

PROBE(hfma2_low_broadcast,
    uint32_t bcast;
    asm volatile("prmt.b32 %0, %1, 0, 0x1010;" : "=r"(bcast) : "r"(b));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(bcast), "r"(c));
)

PROBE(hfma2_plain,
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(hfma2_low_highlevel,
    const __half2 av = reinterpret_cast<const __half2 &>(a);
    const __half2 bv = reinterpret_cast<const __half2 &>(b);
    const __half2 cv = reinterpret_cast<const __half2 &>(c);
    __half2 h = __hfma2(av, __half2half2(__low2half(bv)), cv);
    r = reinterpret_cast<const uint32_t &>(h);
)

PROBE(hfma2_low_mov_ptx,
    asm volatile("{ .reg .b16 lo, hi; .reg .b32 bc; "
                 "mov.b32 {lo,hi}, %2; mov.b32 bc, {lo,lo}; "
                 "fma.rn.f16x2 %0, %1, bc, %3; }"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(bitplane_and_popc,
    r = __popc(a & b) + c;
)

PROBE(lop3_select,
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xca;"
                 : "=r"(r) : "r"(a), "r"(b), "r"(c));
)

PROBE(ternary_sad_masked,
    // a: four unsigned activation bytes. b: byte mask (0x00 or 0xff).
    // Gives sum of the selected activation bytes, plus c.
    const uint32_t masked = a & b;
    asm volatile("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
                 : "=r"(r) : "r"(masked), "r"(0u), "r"(c));
)
