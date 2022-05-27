#!/usr/bin/make

# CUDA compiler
NVCC = nvcc
# NVCC_FLAGS = -g -Wno-deprecated-gpu-targets
NVCC_FLAGS = -Wno-deprecated-gpu-targets -O3

default:
	$(NVCC) $(NVCC_FLAGS) bfs.cu -lcudart -o bfs.exe

prefetch:
	$(NVCC) $(NVCC_FLAGS) -DPREF bfs.cu -lcudart -o bfs.exe

clean:
	rm -f bfs.exe *.linkinfo *.txt gpgpusim_power_report* _cuobjdump_*
