# GPU2/3 continuation

Latest user authorization expands GPU access to GPU2/3. This supersedes the
GPU0-only wording in the earlier goal record. GPU1 remains excluded.

- Parent W4A16: GPU2 UUID GPU-2aa85c85-bc04-0ac3-fc1b-4827d8303d81.
- Dalton W4A4: GPU3 UUID GPU-4868830a-c1cf-90bd-8018-2360c55293b8.
- Sartre: independent CPU-only implementation/accuracy review.

The existing global benchmark lock remains authoritative. Separate devices
do not justify bypassing another session's compiler/timing coordination.
Parent queue session53484 uses token w4a16-long-prefill-20260908-gpu2-r3;
GPU0-r2 queue85210 expired without acquisition. Claims are conditional until
the supervisor prints HELD. A handoff request is recorded in root coordination.

Rebuild before GPU execution: build/worker.cu is newly prepared, while binary
and manifest are still V3. This mismatch is intentional and must not be bypassed.
First test is finite-scale/code exhaustive packed dequantization under memcheck,
then wide kernel smoke and matched gate/up/down screens. All GEMM accumulation
remains FP32. Fresh T64 controls are required on each physical device.

CPU checks rerun after migration: independent W4A4 oracle 6 tests PASS;
W4A16 wide numerical/address model 3 tests PASS. No new GPU result yet.
SASS whitelist tightened to require PACKED=true specifically, rather than
matching either boolean template argument. No production modifications.

Additional correctness-only mode prepared: --scale-stress 1 filters to current
T64, exact Q4 specialization and split-rail wide kernels. It uses signed half
scales with raw magnitudes0x1800..0x37ff, a half-rounded weight CPU oracle and
full bitwise comparison to current T64. It exits before timing. The CPU model
checks that these scales really exercise weight-rounding boundaries; all four
wide-model tests PASS. This mode is not yet compiled or GPU-validated.
