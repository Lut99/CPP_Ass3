debug		:= 0
include		./Makefile.inc


all: clean encrypt checksum

encrypt: encrypt.cu
	$(NVCC) -c encrypt.cu $(NVCCFLAGS) $(INCLUDES)
	$(LINKER) -o $(PROJ_BASE)/encrypt encrypt.o timer.cc $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)

checksum: checksum.cu
	$(NVCC) -c checksum.cu $(NVCCFLAGS) $(INCLUDES)
	$(LINKER) -o $(PROJ_BASE)/checksum checksum.o timer.cc $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)


clean:
	rm -f ./encrypt.o ./checksum.o
	rm -f $(PROJ_BASE)/encrypt $(PROJ_BASE)/checksum
