# Wider compressed-Q4 screen: closed as a performance lead

GPU2, 2026-09-08, snapshot v4. All kernels retain FP32 accumulation and charge
activation conversion inside GEMM. Current compressed-Q4 T64 control is unchanged.
Seven repetitions, eight iterations,2s warmup,64 equal experts,M256,grid280
candidate /112 current control. These are screening shapes, not trace replay.

| Candidate | Projection | Current T64 µs | Candidate µs | Ratio |
|---|---|---:|---:|---:|
| wide packed split U32 | gate/up |7045.116|5579.572|1.263×|
| wide packed sequential U32 | gate/up |7044.548|5663.136|1.244×|
| wide packed split U32 | down |6796.336|6391.732|1.063×|

The existing smaller fused lead remains~4954µs gate/up,~5060µs down on these
fixtures. Wider kernels therefore lose to it. Split kernels use253 registers;
sequential U32 uses187; all24KiB shared,no spills. No claim that register count
alone proves the bottleneck. No larger sweep without a changed hypothesis.

Correctness evidence under gpu-results/:

- gpu2-wide-dequant-check:1,015,808 finite half-scale/code pairs, zero mismatches,
  zero memcheck errors. Paired-half dequantization matches half-rounded Q4_0.
- gpu2-wide-split-smoke:8448 outputs, full CPU oracle and bit identity to split
  control; zero memcheck errors.
- gpu2-wide-scale-stress:16640 outputs bit-identical to current T64 with signed
  non-dyadic half scales and activation-range stress;256 CPU oracle checks;
  zero memcheck errors. Requested wide_packed_split_u32 is present in output.
- gpu2-wide-packed-smoke:8448 outputs match sequential FP32 reference;
  zero memcheck errors. Sequential FP32 differs from T64's split order.

Independent review identified invalid CLI combinations that could falsely run
only controls. Actual requests above used separate dequant/stress modes and
explicitly tested the split candidate, so this does not invalidate them. Fix
the argument/filter guard before the next build. Current stress is correctness
only and never emits timing samples.

Important precision distinction: older delivery_fused_u32 forms q*d in FP32
without current T64's half-rounded weight operand. Its easy fixture is exact;
it is not a real-scale byte-identity proof. Next lead: smaller tile plus the
hardware-verified paired half weight decoder and current split FP32 order.
Model accuracy remains byteidentity or paired PPL±.003, not synthetic rel-L2.

No1.5× goal achieved, no model gain inferred. GPU2 released normally with5MiB,
ECC0; release-1788902129011291688.json. No production edits/commits/pushes.
