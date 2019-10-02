###############################################################
# CUDA_HOME are supposed to be on default position
# and set it in your PATH .bashrc
###############################################################
INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcufft_static -lculibos -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3  -arch=sm_70 --ptxas-options=-v --use_fast_math -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = CONV.exe

ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

all: clean analyze

analyze: CONV_C2C.o CONV-32bit_cuFFT.o Makefile
	$(NVCC) -o $(ANALYZE) CONV-32bit_cuFFT.o CONV_C2C.o $(LIB) $(NVCCFLAGS) 

CONV-32bit_cuFFT.o: timer.h utils_cuda.h
	$(NVCC) -c CONV-32bit_cuFFT.cu $(NVCCFLAGS) -dc -m64

CONV_C2C.o: CONV_C2C.cpp
	$(GCC) -c CONV_C2C.cpp $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


