# W4A16 precision-preserving kernel study

- Input weights use Q4_0's native 18-byte G32 blocks (half scale + 16 bytes).
  Lossless nibble/bit-plane repacks retain the same codes/scales/payload size.
- Activations remain their original binary16 values. Load as 16-bit words,
  widen exactly to binary32; no activation quantizer and no half arithmetic.
- Compare native one-warp-per-row direct arithmetic, its lossless repacked
  counterpart, tiled row-parallel direct arithmetic and register-LUT. Keep
  matching nibble/plane direct controls. No half2 scheme from closed research.
- Main strict candidate uses 32 K stripes and the same FP32 reduction tree as
  the baseline. Other split counts and LUT are marked as reordering variants.
  Verify CPU operation-order references, bitwise baseline equality where
  required, and errors against an exact integer/rational dot oracle.
- First CPU/exhaustive conversion + layout tests and SM60 SASS/spill audit;
  then reserve GPU1 only, validate tails/edges, run sanitizer smoke, sweep
  bounded reuse/split settings, freeze and repeat across four shapes/3 workers.
- Charge final reductions/conversions/table construction. Log CPU repacking
  separately. This is a native-format kernel study, not stock llama.cpp or
  end-to-end model throughput/quality validation. No production integration.
- Hold the single-GPU reservation through all GPU tests, gaps and final checks;
  fresh worker per run, stop on error/hang, no resets or unknown encodings.
- Baseline-strengthening control: repeat each mapping with a lossless FP16 to
  FP32 activation-preparation kernel, included in every timed pipeline. This
  uses FP32 scratch (4*N*K bytes), not activation quantization. Compare matched
  controls before claiming gains. Initial artifacts remain archived separately.
