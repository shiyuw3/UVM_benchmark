include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = 3D.cu

EXE = 3D

OUTPUT = *.out

FLAGS = #-g -G #-arch sm_20 --ptxas-options=-v
release: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) $(SRC) -o $(EXE) -lcudart -Wno-deprecated-gpu-targets -O3

clean: $(SRC)
	rm -f *~ gpgpusim_power_report* _cuobjdump_* *.txt
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT)
