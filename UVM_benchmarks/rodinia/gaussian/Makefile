include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	# $(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -lcudart -Wno-deprecated-gpu-targets

clean:
	rm gaussian
	rm -f *~ gpgpusim_power_report* _cuobjdump_* *.txt
