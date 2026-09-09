2026-09-09: Ported the W16 dual-rail kernel to the serving F32 wire by using the
selected W8 in-kernel F32-to-FP16 loader unchanged. Three fresh paired GPU2
workers per M256 projection give W16 2528.34 vs W8 2589.49 us gate/up
(1.0243x), and W16 2411.37 vs W8 2455.98 us down (1.0190x). Every worker
favors W16. All 127,076,352 output words match W8 bit for bit; 4,608 sampled
CPU dots, memcheck and racecheck pass. Synthetic only; native-model PPL/KLD
and full-model performance remain unqualified. No production edit, commit,
push or reset.
