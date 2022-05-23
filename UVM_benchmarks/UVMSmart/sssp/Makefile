#!/usr/bin/make

default:
	nvcc -O3 -Wno-deprecated-gpu-targets sssp.cu -lcudart -o sssp.exe

prefetch:
	nvcc -g -DPREF -Wno-deprecated-gpu-targets sssp.cu -lcudart -o sssp.exe

clean:
	rm -rf sssp.exe gpgpusim_power_report* _cuobjdump_* *.txt
