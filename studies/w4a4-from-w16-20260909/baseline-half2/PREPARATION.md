# CPU/compiler preparation result

No GPU was queried or executed for this baseline.

The independent CPU suite passes 256 packed-byte decodes, 3,200,000 bounded
G32 prefix transitions, all 64 rows of the word-major tiled-weight layout and
10,000 packed activation groups. The SM60 executable compiles with CUDA 12.8
and g++ 13 without fast math.

Static results for the two serving-shape specializations:

| Kernel | Registers | Shared | Instructions | HFMA2 | FFMA | LDL/STL |
|---|---:|---:|---:|---:|---:|---:|
| W16 pair, either shape | 126 | 32,768 B | 1,872 | 1,024 | 0 | 0/0 |
| Exact-G32 W4 pair, down | 144 | 34,816 B | 2,130 | 1,024 | 64 | 0/0 |
| Exact-G32 W4 pair, gate/up | 144 | 34,816 B | 2,136 | 1,024 | 64 | 0/0 |

The W4 activation quantizer uses 20 registers, no shared/local memory, and no
spills. Generic W4 pair fallback compilation uses 153 registers. The same
M128 body is used for pair and leftover lists, so no separate W4 fallback
kernel is required.

Compressed global weight delivery falls from 4,096 bytes to 1,152 bytes per
M64/K32 stage. Packed activation delivery is 16 code bytes plus a four-byte
FP32 scale per row/G32, versus 64 bytes of FP16 values. These reductions do
not lower the mainloop's static 1,024 HFMA2 count. Exactness also introduces
the 64 per-group FP32 scale FMAs and raw-dot widening represented above, plus
the separately timed quantizer. Static compilation therefore provides no 2x
claim; the prepared GPU comparison must measure the complete pipeline.

`manifest.json` freezes all inputs and generated artifacts. `resources.txt`,
`worker.sass`, and `sass-audit.json` contain the compiler evidence.
