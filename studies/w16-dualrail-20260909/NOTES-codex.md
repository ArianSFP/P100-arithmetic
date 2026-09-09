2026-09-09: Built a W16A16 version of the fastest confirmed W8A16
FP16-storage pipeline by replacing only its paired Q8 decoder with vectorized
FP16 weight loads. Three fresh paired workers per M256 projection show W16
2462.50 vs W8 2514.45 us gate/up (1.0214x) and W16 2353.96 vs W8 2393.43 us
down (1.0175x). W16 is byte-identical to W8 on 242,419,712 checked output
words; 7,680 sampled CPU dots, memcheck and racecheck pass. Specialized W16
uses 126 registers/32KiB shared/zero spills. Synthetic kernel result only;
native-model PPL/KLD and full-model performance remain unqualified. No
production edit, commit, push or reset.

Follow-up single-worker screens: W16/W8 is 0.9362x/1.0232x at M64 GU/down,
1.0224x/1.0185x at M128, and 1.0253x/1.0212x at M512. M64 GU is fallback-only
and remains a 6.81% loss; every screen with M128 pairs favors W16. Including
these screens, 242,419,712 output words match exactly and 7,680 sampled CPU
dots pass. See `summary.json` and `RESULTS.md`.
