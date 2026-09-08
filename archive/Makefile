CUDA ?= /usr/local/cuda-12.8
NVCC := $(CUDA)/bin/nvcc
HOSTCXX ?= /usr/bin/g++-14
NVCC_HOST := -ccbin $(HOSTCXX)

CXXFLAGS := -O2 -std=c++17

.PHONY: all clean inspect

all: build/probe.cubin build/hfma2-worker build/merge_probe.cubin build/merge-worker build/elf_inspect.py build/patch_cubin.py build/enumerate_selectors.py build/oracle.py build/run_candidate.py build/disassemble_candidates.py

build/probe.cubin: src/probe.cu | build
	$(NVCC) $(NVCC_HOST) -O3 -arch=sm_60 -cubin $< -o $@

build/hfma2-worker: src/worker.cpp | build
	$(NVCC) $(NVCC_HOST) $(CXXFLAGS) $< -o $@ -lcuda

build/merge_probe.cubin: src/merge_probe.cu | build
	$(NVCC) $(NVCC_HOST) -O3 -arch=sm_60 -cubin $< -o $@

build/merge-worker: src/merge_worker.cpp | build
	$(NVCC) $(NVCC_HOST) $(CXXFLAGS) $< -o $@ -lcuda

build/elf_inspect.py: scripts/elf_inspect.py | build
	cp $< $@
	chmod +x $@

build/patch_cubin.py: scripts/patch_cubin.py | build
	cp $< $@
	chmod +x $@

build/enumerate_selectors.py: scripts/enumerate_selectors.py | build
	cp $< $@
	chmod +x $@

build/oracle.py: scripts/oracle.py | build
	cp $< $@
	chmod +x $@

build/run_candidate.py: scripts/run_candidate.py | build
	cp $< $@
	chmod +x $@

build/disassemble_candidates.py: scripts/disassemble_candidates.py | build
	cp $< $@
	chmod +x $@

build:
	mkdir -p $@

clean:
	find build -maxdepth 1 -type f ! -name '.*' -delete
