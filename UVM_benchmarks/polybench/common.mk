all:
	nvcc -O3 ${CUFILES} ${DEF} -o ${EXECUTABLE} -lcudart -Wno-deprecated-gpu-targets
clean:
	rm -f *~ *.exe gpgpusim_power_report* _cuobjdump_* *.txt
