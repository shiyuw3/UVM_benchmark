include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

all: kmeans_cuda kmeans_standard

kmeans_cuda:
	$(CC) -std=c++11 kmeans_cuda.cu -o kmeans_cuda -lcudart -Wno-deprecated-gpu-targets -O3

kmeans_standard:
	$(CC) -std=c++11 standard_kmeans.cu -o kmeans_standard -Wno-deprecated-gpu-targets -O3

clean:
	rm -f kmeans_cuda kmeans_standard
	rm -f *~ gpgpusim_power_report* _cuobjdump_* *.txt
