include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

EXE = needle

release: $(SRC)
	$(CC) -O3 ${KERNEL_DIM} $(SRC) -o $(EXE) -lcudart -Wno-deprecated-gpu-targets

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
	rm -f *~ gpgpusim_power_report* _cuobjdump_* *.txt
