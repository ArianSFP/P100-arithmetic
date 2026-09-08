# Four-token P100 decode optimization

See [RESULTS.md](RESULTS.md) for measured comparisons, implementation, validation and accuracy limits. The standalone FP16 candidate reaches 1.48–1.52× over original FP32, or 1.31–1.34× over newly improved byte-identical FP32 controls. Not production qualified.

Build with `python3 build.py`. GPU execution requires a free device, coordination and the guarded controller; saved jobs describe reproducible inputs. Raw results, four source/binary revisions, sanitizer output and SHA-256 provenance are retained. No model integration is included.
