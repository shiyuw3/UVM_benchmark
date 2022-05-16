# NVCCCFLAGS =  -std=c++11 -lcuda -O0
NVCCCFLAGS =  -std=c++11 -lcuda -O3
NVCC = /usr/local/cuda/bin/nvcc

knn: knn_cuda.cu
	$(NVCC) ${NVCCCFLAGS} -o $@ $< -lcudart -Wno-deprecated-gpu-targets

clean:
	rm -f knn
	rm -f *~ gpgpusim_power_report* _cuobjdump_* *.txt
