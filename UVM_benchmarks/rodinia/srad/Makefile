include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = srad.cu

EXE = srad

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -lcudart -Wno-deprecated-gpu-targets -O3

clean: $(SRC)
	rm -f *~ gpgpusim_power_report* _cuobjdump_* *.txt
	rm -f $(EXE) $(EXE).linkinfo result.txt
