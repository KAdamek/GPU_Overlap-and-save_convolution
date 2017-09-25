###############################################################
# CUDA_HOME are supposed to be on default position
# and set it in your PATH .bashrc
###############################################################
INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_60 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = CONV.exe


ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

all: clean onefilter twofilters

onefilter: CONV_1f.o  CONV-32bit_AAFFT-T.o Makefile
	$(NVCC) -o CONV_1f.exe CONV_1f.o CONV-32bit_AAFFT-T.o $(LIB) $(NVCCFLAGS)
	
twofilters: CONV_2f.o  CONV-32bit_AAFFT-T.o Makefile
	$(NVCC) -o CONV_2f.exe CONV_2f.o CONV-32bit_AAFFT-T.o $(LIB) $(NVCCFLAGS)

CONV-32bit_AAFFT-T.o: timer.h utils_cuda.h
	$(NVCC) -c CONV-32bit_AAFFT-T.cu $(NVCCFLAGS)

CONV_1f.o: CONV_1f.cpp
	$(GCC) -c CONV_1f.cpp $(GCC_OPTS)

CONV_2f.o: CONV_2f.cpp
	$(GCC) -c CONV_2f.cpp $(GCC_OPTS)
	
clean:	
	rm -f *.o *.~ $(ANALYZE)


