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

all: clean onefilter

onefilter: CONV_SM_OLS_C2C.o  CONV-32bit_customFFT.o Makefile
	$(NVCC) -o CONV.exe CONV_SM_OLS_C2C.o CONV-32bit_customFFT.o $(LIB) $(NVCCFLAGS)

CONV-32bit_customFFT.o: timer.h utils_cuda.h
	$(NVCC) -c CONV-32bit_customFFT.cu $(NVCCFLAGS)

CONV_SM_OLS_C2C.o: CONV_SM_OLS_C2C.cpp
	$(GCC) -c CONV_SM_OLS_C2C.cpp $(GCC_OPTS)
	
clean:	
	rm -f *.o *.~ CONV_*.exe


