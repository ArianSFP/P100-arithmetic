# Four-token decode target: 1.4x

Parent study is sealed: ../decode-fp16-20260908. Reuse its verified inputs and controls, never modify its artifacts. User requests a focused N4 performance follow-up.

New evidence: parent N8 winner uses B4, thus doubles grid Y relative to N4. At M4352/R2 the fused N4 grid has only68 CTAs for56 SMs. Coarse K partitioning can raise available parallelism while keeping partial output small, unlike previously rejected global32-stripe reductions.

Partition the original S stripes by residue modulo P, with P=2/4/8 CTAs per row tile. Each CTA reduces S/P stripes locally. A final P-way FP32 tree exactly reproduces the original S-stripe tree: local offsets multiplied by P precede global offsets. Preserve the same FP16 local operations and compare all outputs to an unpartitioned parent anchor. Add an exact sequential FP32 dot variant, requiring byte identity at S32 to the original FP32 reference. This may improve FP32 as well; report both the old and strongest new FP32 denominators.

Target up4352x5120, down5120x4352, square5120x5120; N4. Measure preparation plus all reduction/launch costs. Freeze after screen, use three fresh rotated paired workers per shape plus isolated arms. Keep large-range/cancellation/subnormal diagnostics and memory/sync checks. FP16 still requires the model PPL/KLD gate; matching the old FP16 path is not model qualification. No automatic deployment, commit, push, GPU reset or retry after failure.
