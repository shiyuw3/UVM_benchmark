#!/usr/bin/make

default:
	nvcc -Wno-deprecated-gpu-targets main.cu -lcudart -o main -O3

clean:
	rm -rf main gpgpusim_power_report* _cuobjdump_* *.txt
